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

/// `out[i] = a[i] + b[i]`. Bindings: a, b, out. Push: {n}.
pub const ADD: Kernel = Kernel {
    shader: "fnn-residual",
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

pub const ALL: &[Kernel] = &[RELU, ADD, LINEAR];
