//! Every kernel grilly-torch launches. The only place shader names appear.
//! Binding order and push layout are taken from each shader's GLSL source.

use grilly_core::Kernel;

/// `out[i] = max(0, x[i])`. Bindings: x, out. Push: {n}.
pub const RELU: Kernel = Kernel {
    shader: "activation-relu",
    bindings: 2,
    push_words: 1,
    local_size: [256, 1, 1],
};

/// `grad_in[i] = grad_out[i] * (x[i] > 0)`. Bindings: grad_out, x, grad_in.
/// Push: {n}.
pub const RELU_BACKWARD: Kernel = Kernel {
    shader: "activation-relu-backward",
    bindings: 3,
    push_words: 1,
    local_size: [256, 1, 1],
};

/// `out = x @ W^T (+ b)`, 16x16 tiles; x is (rows, in), W is (out, in).
/// Bindings: x, W, b, out. Push: {rows, in, out, has_bias}.
/// Threads: x = output feature, y = row.
pub const LINEAR: Kernel = Kernel {
    shader: "fnn-linear",
    bindings: 4,
    push_words: 4,
    local_size: [16, 16, 1],
};

/// Linear backward, one pass per push `pass`; each pass writes only its own
/// output, without atomics.
/// Bindings: grad_out, x, W, grad_x, grad_W, grad_b.
/// Push: {rows, in, out, pass}.
/// - pass 0: grad_x = grad_out @ W; threads (in, rows)
/// - pass 1: grad_W = grad_out^T @ x; threads (in, out)
/// - pass 2: grad_b = sum over rows of grad_out; threads (out, 1)
pub const LINEAR_BACKWARD: Kernel = Kernel {
    shader: "fnn-linear-backward",
    bindings: 6,
    push_words: 4,
    local_size: [16, 16, 1],
};

pub const ALL: &[Kernel] = &[RELU, RELU_BACKWARD, LINEAR, LINEAR_BACKWARD];

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
