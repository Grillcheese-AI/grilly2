//! Every kernel grilly-torch launches. The only place shader names appear.
//! Binding order and push layout are taken from each shader's GLSL source.

use grilly_core::Kernel;

/// `out[i] = max(0, x[i])`. Bindings: x, out. Push: {n}.
pub const RELU: Kernel = Kernel {
    shader: "g2-relu",
    bindings: 2,
    push_words: 1,
    local_size: [256, 1, 1],
};

/// `grad_in[i] = grad_out[i] * (x[i] > 0)`. Bindings: grad_out, x, grad_in.
/// Push: {n}.
pub const RELU_BACKWARD: Kernel = Kernel {
    shader: "g2-relu-backward",
    bindings: 3,
    push_words: 1,
    local_size: [256, 1, 1],
};

/// `out = x @ W^T (+ b)`, 16x16 tiles; x is (rows, in), W is (out, in).
/// Bindings: x, W, b, out. Push: {rows, in, out, has_bias}.
/// Threads: x = output feature, y = row.
pub const LINEAR: Kernel = Kernel {
    shader: "g2-linear",
    bindings: 4,
    push_words: 4,
    local_size: [16, 16, 1],
};

/// `out = x @ W^T (+ b)` for at most [`GEMV_MAX_ROWS`] rows — the decode
/// shape. One workgroup per output feature, so the weight row is read once
/// and shared across every input row.
/// Bindings: x, W, b, out. Push: {rows, in, out, has_bias}.
/// Threads: one workgroup of 64 per output feature.
pub const GEMV: Kernel = Kernel {
    shader: "g2-gemv",
    bindings: 4,
    push_words: 4,
    local_size: [64, 1, 1],
};

/// Rows at or below which [`GEMV`] beats the tiled kernel.
///
/// **Two, measured, not the shader's MAX_M of 8.** A first guess of 8 was
/// wrong: the tiled kernel's cost is flat in M up to a full tile, while
/// this one's grows with it, so they cross early. Speedup over the tiled
/// kernel on an RX 6750 XT:
///
/// | shape | M=1 | M=2 | M=4 | M=8 |
/// |---|---|---|---|---|
/// | (512,512) | 1.33x | ~1.00x | 0.80x | 0.64x |
/// | (2048,512) | 1.25x | 1.16x | 0.82x | — |
/// | (4096,1024) | 2.44x | 2.09x | 1.21x | 0.68x |
/// | (127996,512) | 1.69x | 1.40x | 1.11x | 0.79x |
///
/// Two never loses; four loses at small `out_features` and wins at large,
/// which would need a second condition on shape to exploit and is worth
/// nothing to decode — decode is M=1 everywhere. The shader still serves 8
/// so that raising this needs no shader change if a batched-decode shape
/// ever makes the large-N column worth taking.
pub const GEMV_MAX_ROWS: usize = 2;

/// Linear backward, one pass per push `pass`; each pass writes only its own
/// output, without atomics.
/// Bindings: grad_out, x, W, grad_x, grad_W, grad_b.
/// Push: {rows, in, out, pass}.
/// - pass 0: grad_x = grad_out @ W; threads (in, rows)
/// - pass 1: grad_W = grad_out^T @ x; threads (in, out)
/// - pass 2: grad_b = sum over rows of grad_out; threads (out, 1)
pub const LINEAR_BACKWARD: Kernel = Kernel {
    shader: "g2-linear-backward",
    bindings: 6,
    push_words: 4,
    local_size: [16, 16, 1],
};

pub const ALL: &[Kernel] = &[RELU, RELU_BACKWARD, LINEAR, GEMV, LINEAR_BACKWARD];

/// Block-sparse attention forward (g2-flash-fwd.glsl). Bindings: q, k, v,
/// bits, kv_count, kv_list, out, lse. Push: {B, H, L, S, D, Dv, scale, mb,
/// mh, nq, nk, q_off, k_off, v_off}. One workgroup per (query tile, head,
/// batch).
pub const FLASH_FWD: Kernel = Kernel {
    shader: "g2-flash-fwd",
    bindings: 8,
    push_words: 14,
    local_size: [256, 1, 1],
};

/// Block-sparse attention backward, dK and dV (g2-flash-bwd-kv.glsl).
/// Bindings: q, k, v, dout, lse, drow, bits, q_count, q_list, dk, dv. Push:
/// {B, H, L, S, D, Dv, scale, mb, mh, nq, nk, q_off, k_off, v_off, do_off}.
/// One workgroup per (key tile, head, batch).
pub const FLASH_BWD_KV: Kernel = Kernel {
    shader: "g2-flash-bwd-kv",
    bindings: 11,
    push_words: 15,
    local_size: [256, 1, 1],
};

/// Block-sparse attention backward, dQ (g2-flash-bwd-q.glsl). Bindings: q,
/// k, v, dout, lse, drow, bits, kv_count, kv_list, dq. Push: as
/// FLASH_BWD_KV. One workgroup per (query tile, head, batch).
pub const FLASH_BWD_Q: Kernel = Kernel {
    shader: "g2-flash-bwd-q",
    bindings: 10,
    push_words: 15,
    local_size: [256, 1, 1],
};

/// Tile lists from a bool mask (g2-flash-tiles.glsl), in four stages.
/// Bindings: mask, bits, flags, kv_count, kv_list, q_count, q_list. Push:
/// {stage, planes, L, S, nq, nk, mask_off}.
pub const FLASH_TILES: Kernel = Kernel {
    shader: "g2-flash-tiles",
    bindings: 7,
    push_words: 7,
    local_size: [256, 1, 1],
};
