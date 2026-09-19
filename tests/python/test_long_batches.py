"""A single batch of GPU work longer than 2 seconds completes.

v1's `CommandBatch::submit()` is `submitDeferred(); waitForCompletion();`,
and `waitForCompletion()` waits on the fence for at most 2 s (v1's
`kFenceTimeoutNs`), then throws, although the GPU finishes the work. grilly
records work lazily and submits it at the next sync, so any batch longer
than 2 s failed with "vkWaitForFences timed out (VkResult=2)"; dense
attention at S=4096 did. v1 is frozen, so `gc_submit` in the shim now calls
`submitDeferred()`, waits on the queue with `vkQueueWaitIdle` (no timeout),
then lets `waitForCompletion()` see the signalled fence.

The test chains batched matmuls (g2-bmm) with no sync in between, so they
form one batch: 40 steps of (1, 4096, 4096) @ (1, 4096, 4096), measured
about 2.8 s on the RX 6750 XT. Every intermediate stays alive until the
submit, about 2.5 GB. A faster GPU finishes sooner and passes without
exercising the fix; asserting a time floor would only make the test flaky.

What is asserted is correctness: row 0 of the chain, traced in float64 on
the host (a matmul maps each row independently). rtol 1e-3 only catches a
wrong or missing result; it is not an accuracy bound for 40 chained
products.
"""

import numpy as np

import grilly

N = 4096
CHAIN = 40


def test_a_batch_longer_than_two_seconds_completes_correctly():
    rng = np.random.default_rng(1234)
    a_np = rng.standard_normal((1, N, N)).astype(np.float32)
    b_np = (rng.standard_normal((1, N, N)) / np.sqrt(N)).astype(np.float32)  # keeps magnitudes steady
    a, b = grilly.tensor(a_np), grilly.tensor(b_np)
    x = a
    for _ in range(CHAIN):
        x = x @ b  # recorded, not run: no sync until the read below
    got = x[0, 0].numpy()  # the one sync: submits the whole chain

    ref = a_np[0, 0:1].astype(np.float64)
    for _ in range(CHAIN):
        ref = ref @ b_np[0].astype(np.float64)
    np.testing.assert_allclose(got, ref[0], rtol=1e-3, atol=1e-3)
