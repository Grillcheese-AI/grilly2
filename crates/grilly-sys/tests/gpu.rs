//! End-to-end checks of the shim on a real Vulkan device.
//!
//! These exercise the dispatch path, not an op: `activation-relu` is used
//! because it is exact in float32, so every comparison is bit-for-bit
//! (tolerance 0). Shaders come from `GRILLY_SHADER_DIR`, else the
//! `shaders/spv` directory of the v1 checkout the crate was built against.

use grilly_sys::*;
use std::ffi::{c_void, CString};
use std::path::PathBuf;
use std::ptr;

const RELU: &str = "activation-relu";
const WG: u32 = 256; // activation-relu local_size_x

macro_rules! ok {
    ($e:expr) => {{
        #[allow(unused_unsafe)]
        let rc = unsafe { $e };
        assert_eq!(rc, GC_OK, "{} failed: {}", stringify!($e), last_error());
    }};
}

struct Ctx(*mut gc_ctx);

impl Ctx {
    fn new() -> Ctx {
        let mut p = ptr::null_mut();
        ok!(gc_create(&mut p));
        let dir = std::env::var("GRILLY_SHADER_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(V1_DIR).join("shaders/spv"));
        let dir = CString::new(dir.to_str().unwrap()).unwrap();
        let mut n = 0u32;
        ok!(gc_load_shader_dir(p, dir.as_ptr(), &mut n));
        assert!(n > 0, "no shaders found");
        Ctx(p)
    }

    fn caps(&self) -> gc_caps {
        let mut c = unsafe { std::mem::zeroed() };
        ok!(gc_caps_get(self.0, &mut c));
        c
    }

    fn buffer(&self, bytes: usize) -> gc_buf {
        let mut b = 0;
        ok!(gc_buffer_create(self.0, bytes as u64, GC_MEM_DEVICE, &mut b));
        b
    }

    fn upload(&self, buf: gc_buf, offset: usize, data: &[f32]) {
        ok!(gc_upload(self.0, buf, offset as u64, data.as_ptr() as *const c_void, (data.len() * 4) as u64));
    }

    fn download(&self, buf: gc_buf, offset: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        ok!(gc_download(self.0, buf, offset as u64, out.as_mut_ptr() as *mut c_void, (n * 4) as u64));
        out
    }

    fn relu(&self, input: gc_binding, output: gc_binding, n: u32) -> i32 {
        let name = CString::new(RELU).unwrap();
        let bindings = [input, output];
        unsafe {
            gc_dispatch(self.0, name.as_ptr(), bindings.as_ptr(), 2, n.div_ceil(WG), 1, 1,
                        &n as *const u32 as *const c_void, 4)
        }
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { gc_destroy(self.0) }
    }
}

fn whole(buf: gc_buf) -> gc_binding {
    gc_binding { buf, offset: 0, range: 0 }
}

/// Deterministic values in [-1, 1) with both signs and exact zeros.
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            if i % 97 == 0 {
                return 0.0;
            }
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn relu_cpu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// Round `bytes` up to the device's storage-buffer offset alignment.
fn aligned(ctx: &Ctx, bytes: usize) -> usize {
    let a = ctx.caps().min_storage_buffer_offset_alignment.max(4) as usize;
    bytes.div_ceil(a) * a
}

#[test]
fn reports_device() {
    let ctx = Ctx::new();
    let c = ctx.caps();
    let name = unsafe { std::ffi::CStr::from_ptr(c.device_name.as_ptr()) };
    println!(
        "device={:?} coopmat={} fp16={} offset_align={} max_push={}",
        name, c.has_cooperative_matrix, c.has_float16,
        c.min_storage_buffer_offset_alignment, c.max_push_constants_size
    );
    assert!(!name.to_bytes().is_empty());
}

#[test]
fn relu_matches_cpu_exactly() {
    let ctx = Ctx::new();
    let n = 1000; // deliberately not a multiple of the workgroup size
    let x = data(n, 1);
    let (a, b) = (ctx.buffer(n * 4), ctx.buffer(n * 4));
    ctx.upload(a, 0, &x);
    ok!(gc_begin(ctx.0));
    ok!(ctx.relu(whole(a), whole(b), n as u32));
    ok!(gc_submit(ctx.0));
    assert_eq!(ctx.download(b, 0, n), relu_cpu(&x));
    ok!(gc_buffer_release(ctx.0, a));
    ok!(gc_buffer_release(ctx.0, b));
}

/// v1's descriptor cache keys on (buffer, range) and ignores offset, so two
/// dispatches over the same buffers at different offsets would share a
/// descriptor set. Both halves must come out right in a single batch.
#[test]
fn offset_bindings_do_not_alias() {
    let ctx = Ctx::new();
    let n = 300;
    let off = aligned(&ctx, n * 4);
    let (x0, x1) = (data(n, 2), data(n, 3));

    let staging: Vec<f32> = {
        let mut v = vec![0f32; (off + n * 4) / 4];
        v[..n].copy_from_slice(&x0);
        v[off / 4..off / 4 + n].copy_from_slice(&x1);
        v
    };
    let bytes = staging.len() * 4;
    let (input, output) = (ctx.buffer(bytes), ctx.buffer(bytes));
    ctx.upload(input, 0, &staging);

    let seg = |buf, offset: usize| gc_binding { buf, offset: offset as u64, range: (n * 4) as u64 };
    ok!(gc_begin(ctx.0));
    ok!(ctx.relu(seg(input, 0), seg(output, 0), n as u32));
    ok!(ctx.relu(seg(input, off), seg(output, off), n as u32));
    ok!(gc_submit(ctx.0));

    assert_eq!(ctx.download(output, 0, n), relu_cpu(&x0), "first segment");
    assert_eq!(ctx.download(output, off, n), relu_cpu(&x1), "second segment");
}

#[test]
fn barrier_orders_dependent_dispatches() {
    let ctx = Ctx::new();
    let n = 4096;
    let x = data(n, 4);
    let (a, b, c) = (ctx.buffer(n * 4), ctx.buffer(n * 4), ctx.buffer(n * 4));
    ctx.upload(a, 0, &x);
    ok!(gc_begin(ctx.0));
    ok!(ctx.relu(whole(a), whole(b), n as u32));
    ok!(gc_barrier(ctx.0));
    ok!(ctx.relu(whole(b), whole(c), n as u32));
    ok!(gc_submit(ctx.0));
    assert_eq!(ctx.download(c, 0, n), relu_cpu(&x));
}

#[test]
fn release_during_batch_is_deferred() {
    let ctx = Ctx::new();
    let n = 512;
    let x = data(n, 5);
    let (a, b) = (ctx.buffer(n * 4), ctx.buffer(n * 4));
    ctx.upload(a, 0, &x);
    ok!(gc_begin(ctx.0));
    ok!(ctx.relu(whole(a), whole(b), n as u32));
    ok!(gc_buffer_release(ctx.0, a));
    let _reuse = ctx.buffer(n * 4); // must not be handed a's memory yet
    ok!(gc_submit(ctx.0));
    assert_eq!(ctx.download(b, 0, n), relu_cpu(&x));
    let mut size = 0;
    assert_eq!(unsafe { gc_buffer_size(ctx.0, a, &mut size) }, GC_ERR);
}

#[test]
fn misuse_is_reported_not_thrown() {
    let ctx = Ctx::new();
    let b = ctx.buffer(1024);

    // unknown shader
    let bogus = CString::new("no-such-shader").unwrap();
    ok!(gc_begin(ctx.0));
    let rc = unsafe { gc_dispatch(ctx.0, bogus.as_ptr(), &whole(b), 1, 1, 1, 1, ptr::null(), 0) };
    assert_eq!(rc, GC_ERR);
    assert!(last_error().contains("no-such-shader"), "{}", last_error());

    // transfers while a batch is open
    let x = [1.0f32; 4];
    let rc = unsafe { gc_upload(ctx.0, b, 0, x.as_ptr() as *const c_void, 16) };
    assert_eq!(rc, GC_ERR);
    assert!(last_error().contains("batch is open"), "{}", last_error());

    // same shader, different layout
    ok!(ctx.relu(whole(b), whole(b), 4));
    let name = CString::new(RELU).unwrap();
    let three = [whole(b), whole(b), whole(b)];
    let n = 4u32;
    let rc = unsafe {
        gc_dispatch(ctx.0, name.as_ptr(), three.as_ptr(), 3, 1, 1, 1, &n as *const u32 as *const c_void, 4)
    };
    assert_eq!(rc, GC_ERR);
    assert!(last_error().contains("different binding count"), "{}", last_error());

    // out-of-bounds binding
    let past = gc_binding { buf: b, offset: 0, range: 4096 };
    assert_eq!(ctx.relu(past, whole(b), 4), GC_ERR);
    assert!(last_error().contains("out of bounds"), "{}", last_error());

    ok!(gc_submit(ctx.0));

    // stale handle
    ok!(gc_buffer_release(ctx.0, b));
    assert_eq!(unsafe { gc_buffer_release(ctx.0, b) }, GC_ERR);
}
