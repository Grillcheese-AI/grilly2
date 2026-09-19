#![allow(dead_code)] // shared by several test binaries, each using a subset
//! Reference results from real torch on CPU.
//!
//! Runs the interpreter named by `GRILLY_PYTHON` (default `python`), which
//! must have torch and numpy installed.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const SCRIPT: &str = r#"
import sys, numpy as np, torch
import torch.nn.functional as F

def grads(f, args, g):
    """Gradients of f(*args) w.r.t. every arg, for output gradient g."""
    args = [a.detach().requires_grad_(True) for a in args]
    return torch.autograd.grad(f(*args), args, g, allow_unused=True, materialize_grads=True)

expr, out_path, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
env = {"torch": torch, "F": F, "grads": grads}
for name, path, shape in zip(rest[0::3], rest[1::3], rest[2::3]):
    dims = tuple(int(d) for d in shape.split(",")) if shape else ()
    env[name] = torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(dims))
out = eval(expr, env)
outs = out if isinstance(out, (tuple, list)) else [out]
with open(out_path, "wb") as f:
    for t in outs:
        a = t.detach().contiguous().numpy().astype("<f4").ravel()
        f.write(np.int64(a.size).tobytes())
        f.write(a.tobytes())
"#;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn python() -> String {
    std::env::var("GRILLY_PYTHON").unwrap_or_else(|_| "python".into())
}

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "grilly2-torch-ref-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_f32(path: &PathBuf, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Evaluate `expr` in torch with the named inputs bound as float32 tensors,
/// returning the result flattened row-major.
pub fn torch(expr: &str, inputs: &[(&str, &[f32], &[usize])]) -> Vec<f32> {
    let mut all = torch_many(expr, inputs);
    assert_eq!(all.len(), 1, "`{expr}` returned {} tensors", all.len());
    all.remove(0)
}

/// Like [`torch`] for an expression returning a tuple of tensors.
pub fn torch_many(expr: &str, inputs: &[(&str, &[f32], &[usize])]) -> Vec<Vec<f32>> {
    let dir = scratch_dir();
    let out = dir.join("out.bin");
    let mut cmd = Command::new(python());
    cmd.arg("-c").arg(SCRIPT).arg(expr).arg(&out);
    for (i, (name, data, shape)) in inputs.iter().enumerate() {
        let path = dir.join(format!("in{i}.bin"));
        write_f32(&path, data);
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        cmd.arg(name).arg(&path).arg(dims.join(","));
    }
    let result = cmd.output().unwrap_or_else(|e| panic!("could not run {:?}: {e}", python()));
    if !result.status.success() {
        let about = Command::new(python())
            .args(["-c", "import sys, torch; print(sys.executable, torch.__version__)"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
            .unwrap_or_default();
        panic!(
            "torch reference failed for `{expr}`
interpreter: {about}
{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let bytes = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let mut tensors = Vec::new();
    let mut rest = &bytes[..];
    while !rest.is_empty() {
        let n = i64::from_le_bytes(rest[..8].try_into().unwrap()) as usize;
        let data = &rest[8..8 + n * 4];
        tensors.push(data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        rest = &rest[8 + n * 4..];
    }
    tensors
}

/// Deterministic values in [-1, 1).
pub fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// Largest |a - b|; panics if any element exceeds `atol + rtol * |b|`.
pub fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut max = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= atol + rtol * w.abs(),
            "{what}: element {i}: got {g}, torch {w}, |diff| {d} > atol {atol} + rtol {rtol} * |torch|"
        );
        max = max.max(d);
    }
    max
}

/// Largest |a - b| / bound; panics if any element exceeds its `bound`.
pub fn assert_within(got: &[f32], want: &[f32], bound: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    assert_eq!(got.len(), bound.len(), "{what}: bound length");
    let mut worst = 0f32;
    for (i, ((g, w), b)) in got.iter().zip(want).zip(bound).enumerate() {
        let d = (g - w).abs();
        assert!(d <= *b, "{what}: element {i}: got {g}, torch {w}, |diff| {d} > bound {b}");
        if *b > 0.0 {
            worst = worst.max(d / b);
        }
    }
    worst
}
