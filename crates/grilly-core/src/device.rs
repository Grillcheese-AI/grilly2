use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use grilly_sys as sys;

use crate::error::{Error, Result};

/// Flush the open batch after this many launches, to bound the command
/// buffer and descriptor usage of a long run with no host reads.
const MAX_LAUNCHES_PER_BATCH: u32 = 1024;

/// Memory class of a GPU buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemClass {
    /// VRAM: fastest for kernels; host transfers are staged.
    Device,
    /// Host-visible, write-combined: for data the CPU writes.
    Upload,
    /// Host-visible, cached: for data the CPU reads back.
    Readback,
}

/// What the device can do.
#[derive(Debug, Clone)]
pub struct Caps {
    pub name: String,
    pub cooperative_matrix: bool,
    pub float16: bool,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_push_constants_size: u32,
    pub max_workgroup_count: [u32; 3],
}

struct State {
    open: bool,
    launches: u32,
}

pub(crate) struct Inner {
    ctx: *mut sys::gc_ctx,
    state: Mutex<State>,
    caps: Caps,
    /// Generated kernels (crate::fusion), by GLSL source: compiled once.
    generated: Mutex<std::collections::HashMap<String, &'static str>>,
}

// The shim serialises every call on its own mutex, and `state` serialises
// batch bookkeeping here; the raw context pointer is never exposed.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // Buffers hold an Arc<Inner>, so none are left by now.
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if state.open {
            unsafe { sys::gc_submit(self.ctx) };
        }
        unsafe { sys::gc_destroy(self.ctx) };
    }
}

fn check(rc: i32) -> Result<()> {
    if rc == sys::GC_OK {
        Ok(())
    } else {
        Err(Error::Device(sys::last_error()))
    }
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::Invalid(format!("string contains NUL: {s:?}")))
}

/// A GPU. Cheap to clone; all clones share one Vulkan device.
#[derive(Clone)]
pub struct Device {
    inner: Arc<Inner>,
}

impl Device {
    /// Open the GPU and register the SPIR-V kernels in `shader_dir`. grilly2's
    /// own kernels (the `g2-*` generics) are registered too.
    pub fn open(shader_dir: impl Into<PathBuf>) -> Result<Device> {
        let mut ctx = ptr::null_mut();
        check(unsafe { sys::gc_create(&mut ctx) })?;

        let mut raw: sys::gc_caps = unsafe { std::mem::zeroed() };
        let caps_rc = unsafe { sys::gc_caps_get(ctx, &mut raw) };
        let inner = Inner {
            ctx,
            state: Mutex::new(State { open: false, launches: 0 }),
            caps: Caps {
                name: unsafe { CStr::from_ptr(raw.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
                cooperative_matrix: raw.has_cooperative_matrix != 0,
                float16: raw.has_float16 != 0,
                min_storage_buffer_offset_alignment: raw.min_storage_buffer_offset_alignment,
                max_push_constants_size: raw.max_push_constants_size,
                max_workgroup_count: raw.max_compute_workgroup_count,
            },
            generated: Mutex::new(std::collections::HashMap::new()),
        };
        let device = Device { inner: Arc::new(inner) };
        check(caps_rc)?;

        device.load_kernels(shader_dir.into())?;
        device.load_kernels(PathBuf::from(env!("GRILLY_CORE_SPV_DIR")))?;
        Ok(device)
    }

    /// The registered name of a generated kernel (crate::fusion): compiled
    /// with shaderc and registered in the pipeline cache on first use of
    /// `source`, then looked up by the source text. Names live for the
    /// process (a kernel is compiled once per distinct source).
    pub fn generated_kernel(&self, source: &str) -> Result<&'static str> {
        let mut generated = self.inner.generated.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(name) = generated.get(source) {
            return Ok(name);
        }
        let name: &'static str = Box::leak(format!("g2-fused-{}", generated.len()).into_boxed_str());
        let name_c = cstring(name)?;
        let _state = self.lock();
        check(unsafe {
            sys::gc_compile_shader(self.inner.ctx, name_c.as_ptr(), source.as_ptr().cast(), source.len() as u64)
        })?;
        generated.insert(source.to_owned(), name);
        Ok(name)
    }

    /// Register every `.spv` in `dir`; a later name replaces an earlier one.
    pub fn load_kernels(&self, dir: PathBuf) -> Result<u32> {
        let dir_c = cstring(dir.to_str().ok_or_else(|| {
            Error::Invalid(format!("shader directory is not UTF-8: {}", dir.display()))
        })?)?;
        let mut loaded = 0u32;
        let _state = self.lock();
        check(unsafe { sys::gc_load_shader_dir(self.inner.ctx, dir_c.as_ptr(), &mut loaded) })?;
        if loaded == 0 {
            return Err(Error::Device(format!("no .spv kernels in {}", dir.display())));
        }
        Ok(loaded)
    }

    /// The process-wide default device, opened on first use.
    ///
    /// Kernels are read from `GRILLY_SHADER_DIR` if set, otherwise from the
    /// `shaders/spv` directory of the v1 checkout grilly was built against.
    pub fn default_device() -> Result<Device> {
        static DEFAULT: OnceLock<std::result::Result<Device, Error>> = OnceLock::new();
        DEFAULT
            .get_or_init(|| {
                let dir = std::env::var_os("GRILLY_SHADER_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(sys::V1_DIR).join("shaders").join("spv"));
                Device::open(dir)
            })
            .clone()
    }

    pub fn caps(&self) -> &Caps {
        &self.inner.caps
    }

    pub fn has_kernel(&self, shader: &str) -> Result<bool> {
        let name = cstring(shader)?;
        let mut out = 0u32;
        let _state = self.lock();
        check(unsafe { sys::gc_has_shader(self.inner.ctx, name.as_ptr(), &mut out) })?;
        Ok(out != 0)
    }

    /// Allocate `bytes` of GPU memory. Contents are undefined.
    pub fn alloc(&self, bytes: u64, mem: MemClass) -> Result<Buffer> {
        if bytes == 0 {
            return Err(Error::Invalid("cannot allocate a zero-byte buffer".into()));
        }
        let class = match mem {
            MemClass::Device => sys::GC_MEM_DEVICE,
            MemClass::Upload => sys::GC_MEM_UPLOAD,
            MemClass::Readback => sys::GC_MEM_READBACK,
        };
        let mut handle = 0;
        let _state = self.lock();
        check(unsafe { sys::gc_buffer_create(self.inner.ctx, bytes, class, &mut handle) })?;
        Ok(Buffer { device: self.clone(), handle, bytes, version: AtomicU64::new(0) })
    }

    /// Run everything recorded so far and wait for it.
    pub fn synchronize(&self) -> Result<()> {
        let mut state = self.lock();
        self.flush(&mut state)
    }

    pub(crate) fn same(&self, other: &Device) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn flush(&self, state: &mut State) -> Result<()> {
        if !state.open {
            return Ok(());
        }
        state.open = false;
        state.launches = 0;
        check(unsafe { sys::gc_submit(self.inner.ctx) })
    }

    /// Record one command into the open batch, after a barrier that orders
    /// it behind everything recorded before it.
    fn record_command(&self, command: impl FnOnce(*mut sys::gc_ctx) -> i32) -> Result<()> {
        let mut state = self.lock();
        if state.open {
            check(unsafe { sys::gc_barrier(self.inner.ctx) })?;
        } else {
            check(unsafe { sys::gc_begin(self.inner.ctx) })?;
            state.open = true;
        }
        check(command(self.inner.ctx))?;
        state.launches += 1;
        if state.launches >= MAX_LAUNCHES_PER_BATCH {
            self.flush(&mut state)?;
        }
        Ok(())
    }

    /// Record one kernel launch. The only route from Rust to a dispatch.
    pub(crate) fn record(
        &self,
        shader: &CStr,
        bindings: &[sys::gc_binding],
        groups: [u32; 3],
        push: &[u32],
    ) -> Result<()> {
        self.record_command(|ctx| unsafe {
            sys::gc_dispatch(
                ctx,
                shader.as_ptr(),
                bindings.as_ptr(),
                bindings.len() as u32,
                groups[0],
                groups[1],
                groups[2],
                push.as_ptr() as *const c_void,
                std::mem::size_of_val(push) as u32,
            )
        })
    }

    /// Record a device-side copy of `bytes` between buffers.
    pub(crate) fn record_copy(&self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.record_command(|ctx| unsafe {
            sys::gc_copy(ctx, src.handle, src_offset, dst.handle, dst_offset, bytes)
        })
    }

    /// Record filling `bytes` at `offset` with the 32-bit pattern `word`.
    pub(crate) fn record_fill(&self, dst: &Buffer, offset: u64, bytes: u64, word: u32) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.record_command(|ctx| unsafe { sys::gc_fill(ctx, dst.handle, offset, bytes, word) })
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Device({:?})", self.inner.caps.name)
    }
}

/// GPU memory, released when dropped. Memory still in use by recorded work
/// is returned to the pool only after that work has run.
pub struct Buffer {
    device: Device,
    handle: sys::gc_buf,
    bytes: u64,
    /// Bumped by every in-place write through a tensor; autograd compares it
    /// against the value seen when a tensor was saved for backward.
    version: AtomicU64,
}

impl Buffer {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn len_bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn handle(&self) -> sys::gc_buf {
        self.handle
    }

    pub(crate) fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    pub(crate) fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// Copy `data` into the buffer at byte `offset`. Waits for recorded work.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let mut state = self.device.lock();
        self.device.flush(&mut state)?;
        check(unsafe {
            sys::gc_upload(
                self.device.inner.ctx,
                self.handle,
                offset,
                data.as_ptr() as *const c_void,
                data.len() as u64,
            )
        })
    }

    /// Copy bytes starting at `offset` into `out`. Waits for recorded work.
    pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let mut state = self.device.lock();
        self.device.flush(&mut state)?;
        check(unsafe {
            sys::gc_download(
                self.device.inner.ctx,
                self.handle,
                offset,
                out.as_mut_ptr() as *mut c_void,
                out.len() as u64,
            )
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _state = self.device.lock();
        // Only fails on an unknown handle, which Buffer cannot hold.
        unsafe { sys::gc_buffer_release(self.device.inner.ctx, self.handle) };
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({} bytes)", self.bytes)
    }
}
