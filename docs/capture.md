# Capture and replay

`grilly.graphed` records a callable's GPU commands once and re-issues them
instead of running the Python again. It is the Vulkan analogue of CUDA
graphs, and it exists for one shape of work: a **decode step**, which runs
the same commands over the same buffers every token and changes only the
bytes inside them.

```python
step = grilly.graphed(model.step)
for t in range(tokens):
    y, state = step(x_t, state)
```

## Why it is worth anything

Fusion removes kernel launches. It does not remove the Python that decides
what to launch, and on a decode step that is the larger cost
(`docs/fusion.md`, "Where capture pays"). Replay removes the Python
instead: a recorded command costs a `vkCmd*` call, not a trip through
PyO3, a shape check and an op dispatch.

Measured on an RX 6750 XT, CubbyLLM's `HybridBackbone` decode step through
`torch_alias`, at steady state with the 512-token window full:

| d512/L8 | tok/s | ms/token | dispatches | submits | GPU ms | host ms |
|---|---:|---:|---:|---:|---:|---:|
| eager | 198 | 5.04 | 318 | 0.3 | 2.36 | 2.69 |
| graphed | 326 | 3.07 | 318 | 0.3 | 2.42 | **0.64** |

| d1024/L16 | tok/s | ms/token | dispatches | submits | GPU ms | host ms |
|---|---:|---:|---:|---:|---:|---:|
| eager | 58 | 17.37 | 635 | 0.6 | 11.14 | 6.22 |
| graphed | 79 | 12.68 | 635 | 0.7 | 11.21 | **1.47** |

Same dispatches, same GPU time; the host time is what goes. Per dispatch
that is ~8.5 µs eager against ~2.0 µs replayed. At d1024/L16 the token is
then 89% GPU wait, so replay has taken host cost about as far as it goes
and the next gain has to come from the kernels.

## What a replay can and cannot do

A replay re-issues commands over the *same buffers*. It is correct exactly
when the step is a pure function of the bytes in those buffers.

It cannot reproduce anything the **host** decided. Two things are therefore
refused at capture, loudly, rather than replayed into a stale answer:

- **a read back to the host** — `int(pos)`, `.item()`, `if t > 0`. The
  value decided what ran, and a replay does not run that code;
- **a host write into a tensor** — `torch.tensor(pos + 1)` each step. A
  replay would write the same bytes every time.

Both are the contract CUDA graphs impose too. The fix is the same: keep
the value in a tensor and let device ops advance it. `pos_t + 1` is a
device add; `int(pos_t) + 1` is not.

A constant is not a host write: `full`, `zeros` and `fill_` on a 4-byte
dtype record a device fill and replay correctly. An 8-byte dtype (int64)
materialises its scalar from the host and does not — a float32 counter is
exact to 2²⁴ and avoids it.

The outputs are the **same tensor objects** every replay, overwritten in
place. Read or clone them before the next call.

## The state a capture cannot see

Refusing host reads and writes catches the caller's mistakes. It does not
catch a value that grilly itself put into a recorded command, and there was
one that mattered: `grilly.infer.KVCache` used to keep its fill level on
the host — `append` wrote at a host-computed offset, `attend` pushed the
length to the shader. Capture that and the shapes never change with the
length, so nothing downstream notices; every replay would attend over
exactly the positions the capture saw.

The cache now keeps the level in a **device cursor**: a `(layers,)` buffer
that `g2-kv-write` writes at, `g2-kv-advance` moves on, and the decode
shaders read their length from. A recorded step therefore writes the next
slot and attends over everything held, and a decode loop over the cache is
graphable.

Two consequences worth knowing:

- `get_seq_length` is the **host's mirror**, advanced when the host calls
  `append`. A replay runs no host code, so after replays it lags.
  `device_len` reads the cursor back (one submit) and is the number the
  shaders are acting on.
- a full cache **saturates** under replay rather than writing past its end.
  The host refuses an overrun before it records; a replay has no host to
  refuse, so `g2-kv-write` declines and the cursor stops at `max_len`.

The split count is still a push constant, so it is fixed at capture. The
slices are cut from the device length, so they always cover the cache
however full it is; the count is chosen from `max_len` rather than the
current length, which is what makes it stable — an early step just leaves
the later slices empty and they return at once.

## Keying, and the growing cache

A graph is valid for one set of shapes, so calls are keyed by a **shape
signature** — the shape and dtype of every tensor argument, and the value
of every scalar one, since a scalar can change which work runs.

A signature is recorded on its **second** sighting, not its first. That is
what makes a growing KV cache free: while the cache grows, every token has
a new shape and nothing is ever captured, so the wrapper costs a
dictionary lookup. When the shape settles — a full window, a static cache
— the next token records and every one after replays.

`max_graphs` (default 8) bounds the cache, least-recently-used first. A
graph retains every buffer its commands name, including the step's
intermediates, so an unbounded cache on a step whose shapes keep changing
would hold the entire history.

The counters say which is happening: `.captures`, `.replays`, `.misses`. A
step that never replays is one whose shapes are not settling, and graphing
it is pure overhead.

## Capture is device-wide

grilly has one device and one queue, so a capture sweeps up whatever else
that device is asked to do while it is open — as CUDA's stream capture does
for its stream. Capture from the thread that will replay, with nothing else
running on the GPU. Tests that capture must serialise against each other
for the same reason (`crates/grilly-core/tests/command_graph.rs`).

## Tolerance

Bit-exact, and nothing weaker is acceptable. A replay issues the same
commands over the same buffers as the capture, so not even a summation
order differs. `tests/python/test_graph.py` and
`crates/grilly-core/tests/command_graph.rs` assert `==`.
