//! Kernel fusion (docs/fusion.md): a graph of float32 elementwise ops,
//! compiled into one kernel that keeps every intermediate in registers.
//!
//! The design is grilly v1's `ShaderFusionEngine`: generate GLSL for the
//! graph, compile it at runtime with the shaderc v1 links, register it in
//! v1's pipeline cache, and reuse it by source. The difference is what gets
//! generated: calls to `g2_unary` / `g2_binary` from `g2-common.glsl`, the
//! same functions the unfused `g2-unary` / `g2-binary` kernels run, with
//! every result `precise`. So a fused graph computes exactly what the same
//! ops computed one kernel at a time, bit for bit.
//!
//! Scalars (a learning rate, a dropout scale) are runtime parameters, not
//! literals: a graph compiles once whatever values it later runs with.

use std::fmt::Write as _;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::generic::{broadcast_shape, broadcast_strides, Binary, Unary};
use crate::kernel::{Arg, Kernel};
use crate::tensor::Tensor;

const COMMON: &str = include_str!("../shaders/g2-common.glsl");
const MAX_DIMS: usize = 6;

/// A node of a [`Graph`]; operands are earlier nodes' indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Node {
    /// Input tensor `i`, broadcast to the output shape.
    Input(usize),
    /// Runtime scalar parameter `i`.
    Param(usize),
    /// `op(x; a, b)` with `a`, `b` parameter nodes (0.0 when absent).
    Unary(Unary, usize, Option<usize>, Option<usize>),
    /// `op(x, y)`.
    Binary(Binary, usize, usize),
}

/// An elementwise graph over `inputs` float32 tensors and `params` scalars,
/// with one or more outputs, all of the (broadcast) output shape.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub outputs: Vec<usize>,
    pub inputs: usize,
    pub params: usize,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    fn push(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// The next input tensor.
    pub fn input(&mut self) -> usize {
        self.inputs += 1;
        self.push(Node::Input(self.inputs - 1))
    }

    /// The next scalar parameter.
    pub fn param(&mut self) -> usize {
        self.params += 1;
        self.push(Node::Param(self.params - 1))
    }

    pub fn unary(&mut self, op: Unary, x: usize, a: Option<usize>, b: Option<usize>) -> usize {
        self.push(Node::Unary(op, x, a, b))
    }

    pub fn binary(&mut self, op: Binary, x: usize, y: usize) -> usize {
        self.push(Node::Binary(op, x, y))
    }

    pub fn output(&mut self, node: usize) {
        self.outputs.push(node);
    }

    fn check(&self) -> Result<()> {
        let bad = |what: &str| Err(Error::Invalid(format!("fusion graph: {what}")));
        if self.outputs.is_empty() {
            return bad("no outputs");
        }
        for (k, node) in self.nodes.iter().enumerate() {
            let earlier = |i: usize| i < k;
            let ok = match *node {
                Node::Input(i) => i < self.inputs,
                Node::Param(i) => i < self.params,
                Node::Unary(_, x, a, b) => {
                    earlier(x) && [a, b].iter().flatten().all(|&p| earlier(p) && matches!(self.nodes[p], Node::Param(_)))
                }
                Node::Binary(_, x, y) => earlier(x) && earlier(y),
            };
            if !ok {
                return bad(&format!("node {k} ({node:?}) refers to a later or missing node"));
            }
        }
        if self.outputs.iter().any(|&o| o >= self.nodes.len()) {
            return bad("an output is not a node");
        }
        Ok(())
    }

    /// The generated GLSL. Bindings: 0 metadata, then each input, then each
    /// output. Metadata (uint words): ndim, the shape (MAX_DIMS), per input
    /// its element offset and broadcast strides (1 + MAX_DIMS), then the
    /// parameters' bits. Push constants: the element count.
    pub fn source(&self) -> String {
        let mut s = String::with_capacity(COMMON.len() + 2048);
        s.push_str("#version 450\nlayout(local_size_x = 256) in;\n\n");
        s.push_str(COMMON);
        s.push_str("\nlayout(std430, binding = 0) readonly buffer Meta { uint meta[]; };\n");
        for i in 0..self.inputs {
            let _ = writeln!(s, "layout(std430, binding = {}) readonly buffer In{i} {{ float src{i}[]; }};", 1 + i);
        }
        for o in 0..self.outputs.len() {
            let _ = writeln!(s, "layout(std430, binding = {}) writeonly buffer Out{o} {{ float dst{o}[]; }};", 1 + self.inputs + o);
        }
        s.push_str("layout(push_constant) uniform P { uint n; } p;\n\nvoid main() {\n");
        s.push_str("    uint i = flat_index();\n    if (i >= p.n) return;\n    uint rem = i;\n");
        let base = |j: usize| 1 + MAX_DIMS + j * (1 + MAX_DIMS);
        for j in 0..self.inputs {
            let _ = writeln!(s, "    uint at{j} = meta[{}];", base(j));
        }
        s.push_str("    for (int d = int(meta[0]) - 1; d >= 0; --d) {\n");
        s.push_str("        uint c = rem % meta[1 + d];\n        rem /= meta[1 + d];\n");
        for j in 0..self.inputs {
            let _ = writeln!(s, "        at{j} += c * meta[{} + d];", base(j) + 1);
        }
        s.push_str("    }\n");
        let param_base = base(self.inputs);
        let arg = |p: Option<usize>| p.map_or_else(|| "0.0".to_owned(), |p| format!("t{p}"));
        for (k, node) in self.nodes.iter().enumerate() {
            let _ = match *node {
                Node::Input(j) => writeln!(s, "    precise float t{k} = src{j}[at{j}];"),
                Node::Param(j) => writeln!(s, "    float t{k} = uintBitsToFloat(meta[{}]);", param_base + j),
                Node::Unary(op, x, a, b) => {
                    writeln!(s, "    precise float t{k} = g2_unary({}u, t{x}, {}, {});", op as u32, arg(a), arg(b))
                }
                Node::Binary(op, x, y) => writeln!(s, "    precise float t{k} = g2_binary({}u, t{x}, t{y});", op as u32),
            };
        }
        for (o, &node) in self.outputs.iter().enumerate() {
            let _ = writeln!(s, "    dst{o}[i] = t{node};");
        }
        s.push_str("}\n");
        s
    }

    /// Run the graph: `inputs` (float32, each broadcastable to `shape`) and
    /// `params` in, one new tensor of `shape` per output out.
    pub fn run(&self, inputs: &[&Tensor], params: &[f32], shape: &[usize]) -> Result<Vec<Tensor>> {
        self.check()?;
        if inputs.len() != self.inputs || params.len() != self.params {
            return Err(Error::Invalid(format!(
                "fusion graph takes {} inputs and {} params, got {} and {}",
                self.inputs,
                self.params,
                inputs.len(),
                params.len()
            )));
        }
        let first = inputs.first().ok_or_else(|| Error::Invalid("fusion graph without inputs".into()))?;
        let device = first.device();
        if shape.len() > MAX_DIMS {
            return Err(Error::NotImplemented(format!("fused kernels over more than {MAX_DIMS} dimensions")));
        }
        for t in inputs {
            if t.dtype() != DType::F32 {
                return Err(Error::Invalid(format!("fused kernels take float32 inputs, got {}", t.dtype().name())));
            }
            if broadcast_shape(t.shape(), shape)? != shape {
                return Err(Error::Shape(format!("input of shape {:?} does not broadcast to {shape:?}", t.shape())));
            }
        }
        let outputs = self.outputs.iter().map(|_| Tensor::empty(device, shape)).collect::<Result<Vec<_>>>()?;
        let n: usize = shape.iter().product();
        if n == 0 {
            return Ok(outputs);
        }
        let word = |v: usize| u32::try_from(v).map_err(|_| Error::Shape(format!("{v} exceeds u32")));
        let mut meta = vec![shape.len() as u32];
        for d in 0..MAX_DIMS {
            meta.push(word(shape.get(d).copied().unwrap_or(1))?);
        }
        for t in inputs {
            meta.push(word(t.offset())?);
            let strides = broadcast_strides(t.shape(), shape);
            for d in 0..MAX_DIMS {
                meta.push(word(strides.get(d).copied().unwrap_or(0))?);
            }
        }
        meta.extend(params.iter().map(|p| p.to_bits()));
        let bytes: Vec<u8> = meta.iter().flat_map(|w| w.to_le_bytes()).collect();
        let meta = Tensor::from_bytes(device, &bytes, &[meta.len()], DType::I32)?;

        let kernel = Kernel {
            shader: device.generated_kernel(&self.source())?,
            bindings: (1 + inputs.len() + outputs.len()) as u32,
            push_words: 1,
            local_size: [256, 1, 1],
        };
        let mut args = vec![Arg::Tensor(&meta)];
        args.extend(inputs.iter().map(|t| Arg::Storage(t)));
        args.extend(outputs.iter().map(Arg::Tensor));
        kernel.launch_flat(device, &args, n, &[word(n)?])?;
        Ok(outputs)
    }
}

// ── the backward of a graph ─────────────────────────────────────────────

/// A graph's backward, as another elementwise graph ([`Graph::backward`]).
#[derive(Debug, Clone)]
pub struct Backward {
    /// Inputs: the forward graph's inputs, then the upstream gradient.
    pub graph: Graph,
    /// Its parameters: the forward's, then constants the rules need.
    pub params: Vec<f32>,
    /// For each forward input, the backward output holding its gradient (at
    /// the output shape; the caller sums it down to the input's shape), or
    /// None when no path reaches it (a zero gradient).
    pub grads: Vec<Option<usize>>,
}

impl Graph {
    /// Reverse mode through output `out` for the inputs flagged in `wanted`,
    /// with `params` the values the forward runs with. The rules are the
    /// eager backward formulas of grilly_torch::math term for term, so the
    /// gradients match eager up to the order contributions are summed in.
    /// Values a rule needs (y = tanh(x), say) are recomputed in the kernel.
    pub fn backward(&self, out: usize, params: &[f32], wanted: &[bool]) -> Result<Backward> {
        self.check()?;
        if out >= self.nodes.len() || params.len() != self.params || wanted.len() != self.inputs {
            return Err(Error::Invalid("fusion backward: bad output, params or wanted flags".into()));
        }
        // which nodes depend on a wanted input
        let mut reaches = vec![false; self.nodes.len()];
        for (k, node) in self.nodes.iter().enumerate() {
            reaches[k] = match *node {
                Node::Input(i) => wanted[i],
                Node::Param(_) => false,
                Node::Unary(_, x, _, _) => reaches[x],
                Node::Binary(_, x, y) => reaches[x] || reaches[y],
            };
        }
        let forward = Graph { nodes: self.nodes[..=out].to_vec(), outputs: vec![], inputs: self.inputs + 1, params: self.params };
        let mut b = Rules { g: forward, params: params.to_vec(), constants: Vec::new() };
        let grad = b.g.push(Node::Input(self.inputs));
        let mut adj: Vec<Option<usize>> = vec![None; out + 1];
        adj[out] = Some(grad);
        let mut input_grad: Vec<Option<usize>> = vec![None; self.inputs];
        for k in (0..=out).rev() {
            let Some(gk) = adj[k] else { continue };
            if !reaches[k] {
                continue;
            }
            match self.nodes[k] {
                Node::Input(i) => input_grad[i] = Some(b.sum(input_grad[i], gk)),
                Node::Param(_) => {}
                Node::Unary(op, x, a, pb) => {
                    if let Some(c) = b.unary_rule(op, gk, x, k, a, pb) {
                        adj[x] = Some(b.sum(adj[x], c));
                    }
                }
                Node::Binary(op, x, y) => {
                    let (cx, cy) = b.binary_rule(op, gk, [x, y, k], reaches[x], reaches[y]);
                    if let Some(c) = cx {
                        adj[x] = Some(b.sum(adj[x], c));
                    }
                    if let Some(c) = cy {
                        adj[y] = Some(b.sum(adj[y], c));
                    }
                }
            }
        }
        let mut grads = vec![None; self.inputs];
        for (i, g) in input_grad.into_iter().enumerate() {
            if let (Some(node), true) = (g, wanted[i]) {
                grads[i] = Some(b.g.outputs.len());
                b.g.outputs.push(node);
            }
        }
        Ok(Backward { graph: b.g, params: b.params, grads })
    }
}

/// Builds a backward graph: the derivative rules, and constants as params.
struct Rules {
    g: Graph,
    params: Vec<f32>,
    constants: Vec<(u32, usize)>,
}

impl Rules {
    fn constant(&mut self, v: f32) -> usize {
        if let Some(&(_, node)) = self.constants.iter().find(|(bits, _)| *bits == v.to_bits()) {
            return node;
        }
        self.params.push(v);
        self.g.params += 1;
        let node = self.g.push(Node::Param(self.g.params - 1));
        self.constants.push((v.to_bits(), node));
        node
    }

    fn un(&mut self, op: Unary, x: usize, a: Option<usize>, b: Option<usize>) -> usize {
        self.g.unary(op, x, a, b)
    }

    fn bin(&mut self, op: Binary, x: usize, y: usize) -> usize {
        self.g.binary(op, x, y)
    }

    fn scalar(&mut self, op: Unary, x: usize, v: f32) -> usize {
        let c = self.constant(v);
        self.un(op, x, Some(c), None)
    }

    /// Adjoint accumulation, in reverse topological order.
    fn sum(&mut self, acc: Option<usize>, c: usize) -> usize {
        match acc {
            None => c,
            Some(a) => self.bin(Binary::Add, a, c),
        }
    }

    fn param_value(&self, node: Option<usize>) -> f32 {
        match node.map(|n| self.g.nodes[n]) {
            Some(Node::Param(i)) => self.params[i],
            _ => 0.0,
        }
    }

    /// The contribution to x of y = op(x; a, b) with upstream g (None: zero).
    fn unary_rule(&mut self, op: Unary, g: usize, x: usize, y: usize, a: Option<usize>, b: Option<usize>) -> Option<usize> {
        let c = match op {
            Unary::Neg => self.un(Unary::Neg, g, None, None),
            Unary::Abs => {
                let s = self.un(Unary::Sign, x, None, None);
                self.bin(Binary::Mul, g, s)
            }
            Unary::Exp => self.bin(Binary::Mul, g, y),
            Unary::Log => self.bin(Binary::Div, g, x),
            Unary::Sqrt => {
                let twice = self.scalar(Unary::MulScalar, y, 2.0);
                self.bin(Binary::Div, g, twice)
            }
            Unary::Rsqrt => {
                let cube = self.scalar(Unary::PowScalar, y, 3.0);
                let m = self.bin(Binary::Mul, g, cube);
                self.scalar(Unary::MulScalar, m, -0.5)
            }
            Unary::Tanh => {
                let sq = self.scalar(Unary::PowScalar, y, 2.0);
                let d = self.scalar(Unary::RsubScalar, sq, 1.0);
                self.bin(Binary::Mul, g, d)
            }
            Unary::Sigmoid => {
                let one_minus = self.scalar(Unary::RsubScalar, y, 1.0);
                let d = self.bin(Binary::Mul, y, one_minus);
                self.bin(Binary::Mul, g, d)
            }
            Unary::AddScalar => g,
            Unary::MulScalar => self.un(Unary::MulScalar, g, a, None),
            Unary::RsubScalar => self.un(Unary::Neg, g, None, None),
            Unary::RdivScalar => {
                let q = self.bin(Binary::Div, y, x);
                let m = self.bin(Binary::Mul, g, q);
                self.un(Unary::Neg, m, None, None)
            }
            Unary::PowScalar => {
                let e = self.param_value(a);
                if e == 0.0 {
                    return None;
                }
                let lower = self.scalar(Unary::PowScalar, x, e - 1.0);
                let m = self.bin(Binary::Mul, g, lower);
                self.un(Unary::MulScalar, m, a, None)
            }
            Unary::Clamp => {
                let mask = self.un(Unary::ClampMask, x, a, b);
                self.bin(Binary::Mul, g, mask)
            }
            Unary::Sin => {
                let c = self.un(Unary::Cos, x, None, None);
                self.bin(Binary::Mul, g, c)
            }
            Unary::Cos => {
                let s = self.un(Unary::Sin, x, None, None);
                let m = self.bin(Binary::Mul, g, s);
                self.un(Unary::Neg, m, None, None)
            }
            Unary::Gelu => {
                let d = self.un(Unary::GeluGrad, x, None, None);
                self.bin(Binary::Mul, g, d)
            }
            // piecewise constant, or masks with no gradient (as eager)
            Unary::Sign
            | Unary::Floor
            | Unary::Ceil
            | Unary::Trunc
            | Unary::Round
            | Unary::ClampMask
            | Unary::GeluGrad
            | Unary::IsFinite
            | Unary::IsInf
            | Unary::IsNan => return None,
        };
        Some(c)
    }

    /// The contributions to x and y of z = op(x, y) with upstream g;
    /// `xyz` is [x, y, z].
    fn binary_rule(&mut self, op: Binary, g: usize, xyz: [usize; 3], need_x: bool, need_y: bool) -> (Option<usize>, Option<usize>) {
        let [x, y, z] = xyz;
        match op {
            Binary::Add => (need_x.then_some(g), need_y.then_some(g)),
            Binary::Sub => (need_x.then_some(g), need_y.then(|| self.un(Unary::Neg, g, None, None))),
            Binary::Mul => (need_x.then(|| self.bin(Binary::Mul, g, y)), need_y.then(|| self.bin(Binary::Mul, g, x))),
            Binary::Div => {
                let ga = self.bin(Binary::Div, g, y);
                let gb = need_y.then(|| {
                    let q = self.bin(Binary::Div, x, y);
                    let m = self.bin(Binary::Mul, ga, q);
                    self.un(Unary::Neg, m, None, None)
                });
                (need_x.then_some(ga), gb)
            }
            Binary::Pow => {
                let ga = need_x.then(|| {
                    let em1 = self.scalar(Unary::AddScalar, y, -1.0);
                    let lower = self.bin(Binary::Pow, x, em1);
                    let d = self.bin(Binary::Mul, y, lower);
                    self.bin(Binary::Mul, g, d)
                });
                // torch zeroes the exponent's gradient where the base is 0
                let gb = need_y.then(|| {
                    let zero = self.constant(0.0);
                    let is_zero = self.bin(Binary::Eq, x, zero);
                    let safe = self.bin(Binary::Add, x, is_zero);
                    let ln = self.un(Unary::Log, safe, None, None);
                    let d = self.bin(Binary::Mul, z, ln);
                    self.bin(Binary::Mul, g, d)
                });
                (ga, gb)
            }
            Binary::Maximum | Binary::Minimum => {
                // the larger (smaller) input gets the gradient; ties split it, as torch
                let win = if op == Binary::Maximum { Binary::Gt } else { Binary::Lt };
                let eq = self.bin(Binary::Eq, x, y);
                let tie = self.scalar(Unary::MulScalar, eq, 0.5);
                let w = self.bin(win, x, y);
                let wa = self.bin(Binary::Add, w, tie);
                let ga = need_x.then(|| self.bin(Binary::Mul, g, wa));
                let gb = need_y.then(|| {
                    let wb = self.scalar(Unary::RsubScalar, wa, 1.0);
                    self.bin(Binary::Mul, g, wb)
                });
                (ga, gb)
            }
            Binary::Eq | Binary::Ne | Binary::Gt | Binary::Ge | Binary::Lt | Binary::Le => (None, None),
        }
    }
}
