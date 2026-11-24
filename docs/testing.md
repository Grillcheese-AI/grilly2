# Testing

Every op has a test that compares it against a reference, and each test
records its tolerance in its header comment. The reference decides the
number, so every test says which of two cases it is in.

## Case (a): grilly against grilly

Examples: a fused region against the same ops recorded eagerly, a compiled
graph against the interpreter, a GPU path against its own fallback. Both
sides run the same kernels in the same framework; only the *order* of the
operations differs.

- **One derivative rule, or one forward op, with no reuse and no
  broadcasting: bit-exact.** `x.to_bits() == y.to_bits()`. The same formula
  runs on the same inputs, so float32 is no excuse for a difference.
- **A composed chain: `rtol = 1e-7`.** This covers chains where reuse or
  broadcasting makes the two sides sum contributions in a different order.
  The bound is there to absorb float32 reassociation and nothing else, and
  `1e-7` is about one ULP of relative error. A difference wider than that is
  a real disagreement that a looser tolerance would disguise.

## Case (b): grilly against torch

Here the two sides are different implementations, with different summation
orders and a different libm. `1e-7` is not reachable, and a test that
demands it has a bug in the test, not in the op.

The per-test numbers are:

- `1e-5` or `1e-6` for single ops;
- `1e-4` after several optimizer steps, because the error compounds through
  the updates.

Each number has its reason recorded beside it.

The preferred form goes further. `crates/grilly-torch/tests/ops_vs_torch.rs`
and `ops_autograd.rs` do this:

1. assert against a float64 reference, under an error bound derived from
   the op;
2. print the worst measured ratio to that bound.

A ratio, not a hand-picked epsilon, is what shows that an op regressed.

## Never widen a tolerance to make a test pass

A wider bound hides a different failure in each case:

- in case (a), a wrong derivative;
- in case (b), a regression the bound would have caught.

Fix the rule instead. If a wider bound really is correct for an op, write
down in the test what makes it correct for that op specifically.

## Where the suites run

CI has no GPU. It runs both suites on Mesa's lavapipe, a software Vulkan
driver. A few tests assert a precision that real GPUs have and Vulkan does
not promise, so lavapipe skips those by name. Each skip comes with its
reason in `.github/workflows/ci.yml`, and the same tests stay strict on
hardware. `notebooks/colab.ipynb` runs the full suite on a Colab NVIDIA
GPU, with nothing skipped.
