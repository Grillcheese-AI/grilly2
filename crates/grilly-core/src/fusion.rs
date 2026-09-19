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
