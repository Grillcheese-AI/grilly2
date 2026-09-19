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
use std::path::PathBuf;
use std::ptr;
use std::time::Instant;

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
    let names = [CString::new("activation-relu").unwrap(), CString::new("activation-tanh").unwrap()];
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
