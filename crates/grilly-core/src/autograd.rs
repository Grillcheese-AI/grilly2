//! Reverse-mode automatic differentiation, following torch's semantics.
//! Design and rationale: docs/autograd.md.
//!
//! An op records a [`Node`] only when grad mode is on and an input requires
//! grad. The graph is owned by the tensors it produced and freed with them.
//! [`backward`] runs nodes highest `seq` first (the reverse of creation
//! order), so gradients are summed in the same order on every run.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use crate::error::{Error, Result};
use crate::ops;
use crate::tensor::Tensor;

// ── grad mode ────────────────────────────────────────────────────────────

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether ops record gradients on this thread (`torch.is_grad_enabled`).
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Restores the previous grad mode when dropped.
#[must_use = "grad mode is restored when the guard is dropped"]
pub struct GradModeGuard {
    previous: bool,
    // Grad mode is per thread; the guard must be dropped where it was made.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.previous));
    }
}

/// Set grad mode on this thread until the guard is dropped
/// (`torch.set_grad_enabled`).
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let previous = GRAD_ENABLED.with(|g| g.replace(enabled));
    GradModeGuard { previous, _not_send: std::marker::PhantomData }
}

/// `torch.no_grad()`.
pub fn no_grad() -> GradModeGuard {
    set_grad_enabled(false)
}

// ── graph ────────────────────────────────────────────────────────────────

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Per-tensor autograd state. Shared by every handle to the same tensor.
pub(crate) struct Meta {
    requires_grad: AtomicBool,
    grad: Mutex<Option<Tensor>>,
    grad_fn: Option<Edge>,
    accumulator: Mutex<Weak<Node>>,
}

impl Meta {
    pub(crate) fn leaf() -> Meta {
        Meta {
            requires_grad: AtomicBool::new(false),
            grad: Mutex::new(None),
            grad_fn: None,
            accumulator: Mutex::new(Weak::new()),
        }
    }

    fn with_grad_fn(edge: Edge) -> Meta {
        Meta { grad_fn: Some(edge), ..Meta::leaf() }
    }
}

#[derive(Clone)]
struct Edge {
    node: Arc<Node>,
    input_nr: usize,
}

/// A step of the recorded graph: how to turn gradients of an op's outputs
/// into gradients of its inputs.
pub struct Node {
    seq: u64,
    name: &'static str,
    /// One per differentiable input; None where that input needs no grad.
    next: Vec<Option<Edge>>,
    num_outputs: usize,
    kind: Kind,
}

enum Kind {
    /// Writes into a leaf's `.grad`. Weak, so the graph never keeps a leaf alive.
    Accumulate { leaf: Weak<Meta>, shape: Vec<usize> },
    Function(Mutex<Box<dyn Backward>>),
}

/// An op's derivative.
pub trait Backward: Send {
    /// Gradients w.r.t. the op's outputs (None: that output got no gradient)
    /// to gradients w.r.t. its inputs, one per input. Entries whose
    /// `needs_input_grad` is false may be None. Runs with grad mode off.
    fn apply(&mut self, grads: &[Option<Tensor>], needs_input_grad: &[bool]) -> Result<Vec<Option<Tensor>>>;

    /// Drop saved tensors; called after `apply` unless the graph is retained.
    fn release(&mut self) {}
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A tensor kept for backward. It is freed with the rest of the graph after
/// `backward()`, and a later use raises torch's "second time" error.
pub struct SavedTensor {
    data: Option<Tensor>,
}

impl SavedTensor {
    pub fn save(t: &Tensor) -> SavedTensor {
        SavedTensor { data: Some(t.detach()) }
    }

    pub fn get(&self) -> Result<&Tensor> {
        self.data.as_ref().ok_or_else(|| {
            Error::Invalid(
                "Trying to backward through the graph a second time (or directly access saved \
                 tensors after they have already been freed). Saved intermediate values of the \
                 graph are freed when you call .backward() or autograd.grad(). Specify \
                 retain_graph=True if you need to backward through the graph a second time or \
                 if you need to access saved tensors after calling backward."
                    .into(),
            )
        })
    }

    pub fn release(&mut self) {
        self.data = None;
    }
}

fn edge_for(t: &Tensor) -> Option<Edge> {
    let meta = t.meta();
    if let Some(edge) = &meta.grad_fn {
        return Some(edge.clone());
    }
    if !meta.requires_grad.load(AtomicOrdering::Relaxed) {
        return None;
    }
    // One accumulator per leaf, so gradients from every use are summed
    // before a single write into `.grad`.
    let mut slot = lock(&meta.accumulator);
    let node = slot.upgrade().unwrap_or_else(|| {
        let node = Arc::new(Node {
            seq: u64::MAX, // run as soon as ready, as torch does
            name: "AccumulateGrad",
            next: Vec::new(),
            num_outputs: 1,
            kind: Kind::Accumulate { leaf: Arc::downgrade(meta), shape: t.shape().to_vec() },
        });
        *slot = Arc::downgrade(&node);
        node
    });
    Some(Edge { node, input_nr: 0 })
}

/// Record an op for autograd. Does nothing, and never calls `make`, unless
/// grad mode is on and some input requires grad. `make` receives the op's
/// outputs, so a formula can save them (see [`SavedTensor`]).
pub fn record<B: Backward + 'static>(
    name: &'static str,
    inputs: &[&Tensor],
    outputs: &mut [&mut Tensor],
    make: impl FnOnce(&[&Tensor]) -> B,
) {
    if !is_grad_enabled() || !inputs.iter().any(|t| t.requires_grad()) {
        return;
    }
    let backward = {
        let outs: Vec<&Tensor> = outputs.iter().map(|t| &**t).collect();
        make(&outs)
    };
    let node = Arc::new(Node {
        seq: SEQ.fetch_add(1, AtomicOrdering::Relaxed),
        name,
        next: inputs.iter().map(|t| edge_for(t)).collect(),
        num_outputs: outputs.len(),
        kind: Kind::Function(Mutex::new(Box::new(backward))),
    });
    for (i, out) in outputs.iter_mut().enumerate() {
        out.set_meta(Arc::new(Meta::with_grad_fn(Edge { node: node.clone(), input_nr: i })));
    }
}

// ── tensor-facing state ──────────────────────────────────────────────────

pub(crate) fn requires_grad(meta: &Meta) -> bool {
    meta.grad_fn.is_some() || meta.requires_grad.load(AtomicOrdering::Relaxed)
}

pub(crate) fn set_requires_grad(meta: &Meta, value: bool) -> Result<()> {
    if meta.grad_fn.is_some() {
        if value {
            return Ok(());
        }
        return Err(Error::Invalid(
            "you can only change requires_grad flags of leaf variables. If you want to use a \
             computed variable in a subgraph that doesn't require differentiation use \
             var_no_grad = var.detach()."
                .into(),
        ));
    }
    meta.requires_grad.store(value, AtomicOrdering::Relaxed);
    Ok(())
}

pub(crate) fn is_leaf(meta: &Meta) -> bool {
    meta.grad_fn.is_none()
}

pub(crate) fn grad_fn_name(meta: &Meta) -> Option<&'static str> {
    meta.grad_fn.as_ref().map(|e| e.node.name)
}

pub(crate) fn grad(meta: &Meta) -> Option<Tensor> {
    lock(&meta.grad).clone()
}

pub(crate) fn set_grad(meta: &Meta, grad: Option<Tensor>) {
    *lock(&meta.grad) = grad;
}

// ── engine ───────────────────────────────────────────────────────────────

struct Ready {
    seq: u64,
    order: u64,
    node: Arc<Node>,
}

impl PartialEq for Ready {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ready {}
impl PartialOrd for Ready {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ready {
    /// Highest seq first; among equal seqs, first-ready first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.seq.cmp(&other.seq).then(other.order.cmp(&self.order))
    }
}

fn key(node: &Arc<Node>) -> usize {
    Arc::as_ptr(node) as usize
}

fn buffer_add(buffers: &mut HashMap<usize, Vec<Option<Tensor>>>, edge: &Edge, grad: Tensor) -> Result<()> {
    let slots = buffers.entry(key(&edge.node)).or_insert_with(|| vec![None; edge.node.num_outputs]);
    let slot = &mut slots[edge.input_nr];
    *slot = Some(match slot.take() {
        None => grad,
        Some(prev) => ops::add(&prev, &grad)?,
    });
    Ok(())
}

fn accumulate_grad(leaf: &Meta, shape: &[usize], grad: Tensor) -> Result<()> {
    if grad.shape() != shape {
        return Err(Error::Shape(format!(
            "gradient of shape {:?} does not match its leaf of shape {:?}",
            grad.shape(),
            shape
        )));
    }
    let mut slot = lock(&leaf.grad);
    match slot.as_ref() {
        // Add in place, so existing references to `.grad` see the sum.
        Some(existing) => ops::add_into(existing, &grad),
        None => {
            // Keep it only if nothing else can see it; otherwise store a copy.
            *slot = Some(if grad.is_sole_owner() { grad.detach() } else { grad.clone_data()? });
            Ok(())
        }
    }
}

/// `root.backward(gradient, retain_graph)`.
pub(crate) fn backward(root: &Tensor, gradient: Option<&Tensor>, retain_graph: bool) -> Result<()> {
    if !root.requires_grad() {
        return Err(Error::Invalid(
            "element 0 of tensors does not require grad and does not have a grad_fn".into(),
        ));
    }
    let root_grad = match gradient {
        Some(g) if g.shape() != root.shape() => {
            return Err(Error::Shape(format!(
                "Mismatch in shape: grad_output[0] has a shape of {:?} and output[0] has a shape of {:?}.",
                g.shape(),
                root.shape()
            )))
        }
        Some(g) => g.detach(),
        None if root.numel() != 1 => {
            return Err(Error::Invalid("grad can be implicitly created only for scalar outputs".into()))
        }
        None => Tensor::full(root.device(), root.shape(), 1.0)?,
    };
    let root_edge = edge_for(root).expect("a tensor that requires grad has an edge");
    let _no_grad = no_grad();

    // How many edges point at each node reachable from the root.
    let mut deps: HashMap<usize, usize> = HashMap::new();
    let mut seen = HashSet::from([key(&root_edge.node)]);
    let mut stack = vec![root_edge.node.clone()];
    while let Some(node) = stack.pop() {
        for edge in node.next.iter().flatten() {
            *deps.entry(key(&edge.node)).or_default() += 1;
            if seen.insert(key(&edge.node)) {
                stack.push(edge.node.clone());
            }
        }
    }

    let mut buffers = HashMap::new();
    buffer_add(&mut buffers, &root_edge, root_grad)?;
    let mut order = 0u64;
    let mut ready = BinaryHeap::from([Ready { seq: root_edge.node.seq, order, node: root_edge.node }]);

    while let Some(Ready { node, .. }) = ready.pop() {
        let grads = buffers.remove(&key(&node)).unwrap_or_else(|| vec![None; node.num_outputs]);
        let input_grads = match &node.kind {
            Kind::Accumulate { leaf, shape } => {
                if let (Some(leaf), Some(Some(g))) = (leaf.upgrade(), grads.into_iter().next()) {
                    accumulate_grad(&leaf, shape, g)?;
                }
                Vec::new()
            }
            Kind::Function(f) => {
                let mut f = lock(f);
                let needs: Vec<bool> = node.next.iter().map(Option::is_some).collect();
                let out = if grads.iter().all(Option::is_none) {
                    vec![None; needs.len()]
                } else {
                    f.apply(&grads, &needs)?
                };
                if !retain_graph {
                    f.release();
                }
                if out.len() != needs.len() {
                    return Err(Error::Invalid(format!(
                        "{} returned {} gradients for {} inputs",
                        node.name,
                        out.len(),
                        needs.len()
                    )));
                }
                out
            }
        };
        for (edge, grad) in node.next.iter().zip(input_grads) {
            let Some(edge) = edge else { continue };
            if let Some(grad) = grad {
                buffer_add(&mut buffers, edge, grad)?;
            }
            let remaining = deps.get_mut(&key(&edge.node)).expect("counted above");
            *remaining -= 1;
            if *remaining == 0 {
                order += 1;
                ready.push(Ready { seq: edge.node.seq, order, node: edge.node.clone() });
            }
        }
    }
    Ok(())
}
