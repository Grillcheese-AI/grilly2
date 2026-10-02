//! Where the GPU time of a dispatch goes inside one batch (CLAUDE.md §11:
//! a decode loop is many tiny dispatches). Drives the shim directly so each
//! factor can be varied alone:
//!
//!   barrier   gc_barrier before every dispatch (what grilly_core records),
//!             or none (valid here only because the dispatches are
//!             independent: each reads one input and writes its own output)
//!   size      elements per dispatch, from 1 (pure fixed cost) up
//!   pipeline  the same kernel every dispatch, or two alternating
//!   chain     each dispatch reading the previous one's output (a barrier
//!             is then required)
//!
//! GPU time per batch is the shim's wait for it (`gc_submit_stats`); host
//! time is the recording loop. Run with
//!   cargo test --release -p grilly-sys --test dispatch_cost -- --ignored --nocapture

use grilly_sys::*;
use std::ffi::{c_void, CString};
use std::ptr;
use std::time::Instant;

const RELU: &str = "probe-relu";
const TANH: &str = "probe-tanh";
const DISPATCHES: usize = 1000; // under the device's 1024-launch batch bound
const OUTPUTS: usize = 16;
const REPEATS: usize = 7;

macro_rules! ok {
    ($e:expr) => {{
        #[allow(unused_unsafe)]
        let rc = unsafe { $e };
        assert_eq!(rc, GC_OK, "{} failed: {}", stringify!($e), last_error());
    }};
}

/// The probe kernel, compiled by the shim itself.
///
/// These tests are about the dispatch path, not about an op, so borrowing
/// a kernel from a shader directory only ever added a thing to find.
/// `gc_compile_shader` is already here for fused kernels; two elementwise
/// one-liners through it leave the crate with no shader files at all and
/// exercise the JIT on the way past. Both are exact in float32, so every
/// comparison below is bit-for-bit.
const PROBE_SRC: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer I { float src[]; };
layout(std430, binding = 1) writeonly buffer O { float dst[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.n) dst[i] = max(src[i], 0.0);
}
"#;

const PROBE2_SRC: &str = r#"#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer I { float src[]; };
layout(std430, binding = 1) writeonly buffer O { float dst[]; };
layout(push_constant) uniform P { uint n; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.n) dst[i] = tanh(src[i]);
}
"#;

fn compile(ctx: *mut gc_ctx, name: &str, src: &str) {
    let n = CString::new(name).unwrap();
    ok!(gc_compile_shader(ctx, n.as_ptr(), src.as_ptr().cast(), src.len() as u64));
}

struct Ctx(*mut gc_ctx);

impl Ctx {
    fn new() -> Ctx {
        let mut p = ptr::null_mut();
        ok!(gc_create(&mut p));
        compile(p, RELU, PROBE_SRC);
        compile(p, TANH, PROBE2_SRC);
        Ctx(p)
    }

    fn buffer(&self, bytes: usize) -> gc_buf {
        let mut b = 0;
        ok!(gc_buffer_create(self.0, bytes as u64, GC_MEM_DEVICE, &mut b));
        b
    }

    fn stats(&self) -> gc_submit_stats {
        let mut s = gc_submit_stats::default();
        ok!(gc_submit_stats_get(self.0, &mut s));
        s
    }
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { gc_destroy(self.0) }
    }
}

#[derive(Clone, Copy)]
struct Config {
    barrier: bool,
    n: u32,
    alternate: bool,
    chain: bool,
}

/// (median GPU us per dispatch, median host us per dispatch)
fn measure(ctx: &Ctx, cfg: Config) -> (f64, f64) {
    let names = [CString::new(RELU).unwrap(), CString::new(TANH).unwrap()];
    let bytes = cfg.n as usize * 4;
    let input = ctx.buffer(bytes);
    let outs: Vec<gc_buf> = (0..OUTPUTS).map(|_| ctx.buffer(bytes)).collect();
    let mut gpu = Vec::new();
    let mut host = Vec::new();
    for rep in 0..REPEATS + 1 {
        ok!(gc_begin(ctx.0));
        let t0 = Instant::now();
        for i in 0..DISPATCHES {
            if cfg.barrier && i > 0 {
                ok!(gc_barrier(ctx.0));
            }
            // a chain reads the previous dispatch's output; otherwise the one input
            let src = if cfg.chain && i > 0 { outs[(i - 1) % OUTPUTS] } else { input };
            let bindings = [
                gc_binding { buf: src, offset: 0, range: 0 },
                gc_binding { buf: outs[i % OUTPUTS], offset: 0, range: 0 },
            ];
            let name = &names[if cfg.alternate { i % 2 } else { 0 }];
            ok!(gc_dispatch(ctx.0, name.as_ptr(), bindings.as_ptr(), 2, cfg.n.div_ceil(256), 1, 1,
                            &cfg.n as *const u32 as *const c_void, 4));
        }
        let recorded = t0.elapsed().as_secs_f64();
        ok!(gc_submit(ctx.0));
        if rep > 0 {
            // the first repeat warms pipelines and descriptor pools
            gpu.push(ctx.stats().wait_ns_last as f64 / 1e3 / DISPATCHES as f64);
            host.push(recorded * 1e6 / DISPATCHES as f64);
        }
    }
    for b in std::iter::once(input).chain(outs) {
        ok!(gc_buffer_release(ctx.0, b));
    }
    gpu.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    (gpu[REPEATS / 2], host[REPEATS / 2])
}

#[test]
#[ignore = "measurement; run with --ignored --nocapture"]
fn dispatch_cost_breakdown() {
    let ctx = Ctx::new();
    println!("{DISPATCHES} dispatches per batch, median of {REPEATS} batches; microseconds per dispatch");
    println!("{:>8} {:>9} {:>10} {:>7} | {:>8} {:>8}", "elements", "barrier", "pipeline", "chain", "GPU us", "host us");
    for n in [1u32, 256, 4096, 65_536, 262_144] {
        for (barrier, alternate, chain) in [
            (true, false, false),
            (false, false, false),
            (true, true, false),
            (false, true, false),
            (true, false, true),
        ] {
            let (gpu, host) = measure(&ctx, Config { barrier, n, alternate, chain });
            println!("{:>8} {:>9} {:>10} {:>7} | {:>8.2} {:>8.2}", n,
                     if barrier { "yes" } else { "no" }, if alternate { "alternate" } else { "same" },
                     if chain { "yes" } else { "no" }, gpu, host);
        }
    }
}
