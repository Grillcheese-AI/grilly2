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
