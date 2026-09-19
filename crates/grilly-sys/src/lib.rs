//! Raw bindings to `shim/gc_shim.h`. See that header for the contract;
//! the safe API lives in `grilly-core`.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void, CStr};

pub const GC_OK: i32 = 0;
pub const GC_ERR: i32 = 1;

pub const GC_MEM_DEVICE: u32 = 0;
pub const GC_MEM_UPLOAD: u32 = 1;
pub const GC_MEM_READBACK: u32 = 2;

#[repr(C)]
pub struct gc_ctx {
    _private: [u8; 0],
}

pub type gc_buf = u64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct gc_caps {
    pub device_name: [c_char; 256],
    pub has_cooperative_matrix: u32,
    pub has_float16: u32,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_push_constants_size: u32,
    pub max_compute_workgroup_count: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct gc_binding {
    pub buf: gc_buf,
    pub offset: u64,
    /// 0 means "to the end of the buffer".
    pub range: u64,
}

extern "C" {
    pub fn gc_last_error() -> *const c_char;

    pub fn gc_create(out: *mut *mut gc_ctx) -> i32;
    pub fn gc_destroy(ctx: *mut gc_ctx);
    pub fn gc_caps_get(ctx: *mut gc_ctx, out: *mut gc_caps) -> i32;

    pub fn gc_load_shader(ctx: *mut gc_ctx, name: *const c_char, spv_path: *const c_char) -> i32;
    pub fn gc_load_shader_dir(ctx: *mut gc_ctx, dir: *const c_char, loaded: *mut u32) -> i32;
    pub fn gc_has_shader(ctx: *mut gc_ctx, name: *const c_char, out: *mut u32) -> i32;

    pub fn gc_buffer_create(ctx: *mut gc_ctx, bytes: u64, mem: u32, out: *mut gc_buf) -> i32;
    pub fn gc_buffer_release(ctx: *mut gc_ctx, buf: gc_buf) -> i32;
    pub fn gc_buffer_size(ctx: *mut gc_ctx, buf: gc_buf, out: *mut u64) -> i32;

    pub fn gc_upload(ctx: *mut gc_ctx, dst: gc_buf, offset: u64, src: *const c_void, bytes: u64) -> i32;
    pub fn gc_download(ctx: *mut gc_ctx, src: gc_buf, offset: u64, dst: *mut c_void, bytes: u64) -> i32;

    pub fn gc_begin(ctx: *mut gc_ctx) -> i32;
    pub fn gc_dispatch(
        ctx: *mut gc_ctx,
        shader: *const c_char,
        bindings: *const gc_binding,
        num_bindings: u32,
        gx: u32,
        gy: u32,
        gz: u32,
        push: *const c_void,
        push_bytes: u32,
    ) -> i32;
    pub fn gc_barrier(ctx: *mut gc_ctx) -> i32;
    pub fn gc_submit(ctx: *mut gc_ctx) -> i32;
}

/// The message for the most recent failing `gc_*` call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(gc_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Root of the grilly v1 checkout this crate was built against.
pub const V1_DIR: &str = env!("GRILLY_SYS_V1_DIR");
