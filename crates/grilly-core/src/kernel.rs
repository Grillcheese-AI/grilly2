use std::ffi::CString;

use grilly_sys as sys;

use crate::device::{Buffer, Device};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A compiled compute kernel and the interface it was written against.
///
/// Every kernel an op uses is declared once as a `Kernel` constant in the
/// crate that owns the op. Ops launch kernels through [`Kernel::launch`] and
/// never name a shader file anywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernel {
    /// SPIR-V module name (file stem).
    pub shader: &'static str,
    /// Number of storage-buffer bindings, in binding order.
    pub bindings: u32,
    /// Push-constant block size in 32-bit words.
    pub push_words: u32,
    /// `local_size_{x,y,z}` declared by the shader.
    pub local_size: [u32; 3],
}

/// One storage-buffer argument to a launch.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    /// A tensor's elements, bound at their byte offset.
    Tensor(&'a Tensor),
    /// A tensor's whole storage buffer; the kernel locates the tensor by an
    /// element offset in its push constants (see `Tensor::offset`).
    Storage(&'a Tensor),
    /// A whole buffer.
    Buffer(&'a Buffer),
}

impl Kernel {
    /// Workgroup counts that cover `threads` invocations in each dimension.
    pub fn groups_for(&self, threads: [u32; 3]) -> [u32; 3] {
        let mut groups = [1u32; 3];
        for d in 0..3 {
            groups[d] = threads[d].div_ceil(self.local_size[d]).max(1);
        }
        groups
    }

    /// Record a launch of one invocation per element for `n` elements, over a
    /// 2-D grid so large `n` stays within dispatch limits. The kernel must
    /// compute its index as `y * num_groups_x * local_x + x` (g2-common.glsl).
    pub fn launch_flat(&self, device: &Device, args: &[Arg<'_>], n: usize, push: &[u32]) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let local = self.local_size[0] as usize;
        let groups = n.div_ceil(local);
        let gx = groups.min(65_535);
        let gy = groups.div_ceil(gx);
        let (gx, gy) = (u32::try_from(gx).unwrap(), u32::try_from(gy).map_err(|_| {
            Error::Shape(format!("{n} elements exceed the dispatch limit"))
        })?);
        self.launch(device, args, [gx * self.local_size[0], gy, 1], push)
    }

    /// Record a launch covering `threads` invocations. Arguments bind in
    /// order to bindings 0..n; `push` is the push-constant block.
    pub fn launch(&self, device: &Device, args: &[Arg<'_>], threads: [u32; 3], push: &[u32]) -> Result<()> {
        if args.len() != self.bindings as usize {
            return Err(Error::Invalid(format!(
                "kernel {} takes {} bindings, got {}",
                self.shader,
                self.bindings,
                args.len()
            )));
        }
        if push.len() != self.push_words as usize {
            return Err(Error::Invalid(format!(
                "kernel {} takes {} push-constant words, got {}",
                self.shader,
                self.push_words,
                push.len()
            )));
        }
        let mut bindings = Vec::with_capacity(args.len());
        for arg in args {
            let (buffer, offset, range) = match arg {
                Arg::Tensor(t) => (t.storage(), t.byte_offset(), t.byte_len()),
                Arg::Storage(t) => (t.storage(), 0, t.storage().len_bytes()),
                Arg::Buffer(b) => (*b, 0, b.len_bytes()),
            };
            if !buffer.device().same(device) {
                return Err(Error::Invalid(format!(
                    "kernel {}: argument lives on a different device",
                    self.shader
                )));
            }
            // A zero range would mean "to the end of the buffer" to the shim.
            bindings.push(sys::gc_binding { buf: buffer.handle(), offset, range: range.max(4) });
        }
        let name = CString::new(self.shader)
            .map_err(|_| Error::Invalid(format!("bad kernel name {:?}", self.shader)))?;
        device.record(&name, &bindings, self.groups_for(threads), push)
    }
}
