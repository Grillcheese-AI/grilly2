//! Fused elementwise regions with autograd (docs/fusion.md): the forward is
//! one kernel (grilly_core::fusion) and so is the backward, a graph derived
//! from the forward by `Graph::backward` with eager's formulas.

use grilly_core::fusion::Graph;
use grilly_core::generic as k;
use grilly_core::{Error, Result, Tensor};

use crate::math::record;

/// The graph's single output over `inputs` and `params`, of `shape`,
/// recorded as one FusedBackward0 node. Its backward runs one fused kernel
/// for every input that needs a gradient, then sums each down to its
/// input's shape.
pub fn fused(graph: &Graph, inputs: &[&Tensor], params: &[f32], shape: &[usize]) -> Result<Tensor> {
    if graph.outputs.len() != 1 {
        return Err(Error::Invalid("a recorded fused region has exactly one output".into()));
    }
    let mut out = graph.run(inputs, params, shape)?.remove(0);
    let (forward, values, out_node) = (graph.clone(), params.to_vec(), graph.outputs[0]);
    let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();
    let shape = shape.to_vec();
    record("FusedBackward0", inputs, &mut out, inputs, false, move |g, saved, needs| {
        let bw = forward.backward(out_node, &values, needs)?;
        let grads = if bw.graph.outputs.is_empty() {
            Vec::new()
        } else {
            let mut ins: Vec<&Tensor> = saved.iter().collect();
            ins.push(g);
            bw.graph.run(&ins, &bw.params, &shape)?
        };
        (0..needs.len())
            .map(|i| {
                if !needs[i] {
                    return Ok(None);
                }
                match bw.grads[i] {
                    Some(o) => k::sum_to_shape(&grads[o], &shapes[i]).map(Some),
                    None => Tensor::zeros(g.device(), &shapes[i]).map(Some),
                }
            })
            .collect()
    })?;
    Ok(out)
}
