"""Capture a step's GPU commands once and replay them (``grilly.graphed``).

The Vulkan analogue of CUDA graphs. A decode step issues the same commands
on the same buffers every token; only the bytes change. Rediscovering that
by running the Python again costs far more than the commands do — measured
on this card, about 10 us of host time per dispatch from Python against
under 2 us to replay a recorded one (docs/capture.md).

    step = grilly.graphed(model.step)
    for t in range(tokens):
        y, state = step(x_t, state)

A call whose **shape signature** has not been seen runs normally. The second
call with that signature is captured, and every later one writes the new
inputs into the tensors the capture read, replays, and returns the tensors
the capture wrote.

Capturing on the *second* sighting rather than the first is what makes a
growing KV cache free: while the cache grows, every token has a new
signature and nothing is ever captured, so the wrapper costs a dictionary
lookup. When the shape settles — a full window, a static cache — the next
token records and the rest replay.

**What a replay can and cannot do.** It re-issues commands over the same
buffers, so it reproduces a step that is a pure function of the bytes in
those buffers. It cannot reproduce anything the *host* decided: a capture
that reads a value back (``int(pos)``, ``.item()``, ``if t > 0``) or writes
one in (``torch.tensor(pos + 1)``) is refused by the device, with a message
saying so, rather than replayed into a stale answer. Keep such values in
tensors and let device ops advance them, which is the same contract CUDA
graphs impose.

The outputs are the *same tensor objects* every replay, overwritten in
place. Read them, or clone them, before the next call.
"""

from __future__ import annotations

import collections
import functools

from . import Tensor, _C

__all__ = ["graphed"]


def _signature(value):
    """A key for the shapes and dtypes of a call's arguments. Two calls with
    the same signature may share a graph; anything else records its own."""
    if isinstance(value, Tensor):
        return ("t", tuple(value.shape), str(value.dtype))
    if isinstance(value, (tuple, list)):
        return ("s", type(value).__name__, tuple(_signature(v) for v in value))
    if isinstance(value, dict):
        return ("d", tuple((k, _signature(v)) for k, v in sorted(value.items())))
    if value is None or isinstance(value, (int, float, bool, str)):
        return ("v", repr(value))  # a scalar can change which work runs, so it keys too
    return ("o", type(value).__name__, id(value))


def _tensors(value, out: list) -> None:
    if isinstance(value, Tensor):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _tensors(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _tensors(item, out)


def _rebuild(value, pool):
    """`value` with every tensor replaced by the next one from `pool`, in the
    order `_tensors` walks them."""
    if isinstance(value, Tensor):
        return next(pool)
    if isinstance(value, tuple):
        return tuple(_rebuild(v, pool) for v in value)
    if isinstance(value, list):
        return [_rebuild(v, pool) for v in value]
    if isinstance(value, dict):
        return {k: _rebuild(v, pool) for k, v in value.items()}
    return value


class _Graphed:
    """One captured graph, the tensors it reads, and the ones it wrote."""

    def __init__(self, inputs: list, outputs, graph) -> None:
        self.inputs, self.outputs, self.graph = inputs, outputs, graph

    def replay(self, new_inputs: list):
        for held, fresh in zip(self.inputs, new_inputs):
            if held is not fresh:
                held.copy_(fresh)  # a device copy; the caller's tensor is untouched
        self.graph.replay()
        return self.outputs


def graphed(fn=None, *, max_graphs: int = 8, enabled: bool = True):
    """Capture ``fn``'s GPU commands per shape signature and replay them.

    Returns a callable with the same signature.

    ``max_graphs`` bounds how many recordings are kept, least-recently-used
    first. A graph holds every buffer its commands name, including the step's
    intermediates, so an unbounded cache on a step whose shapes keep changing
    would retain the whole history. Eight is enough for a decode loop, which
    settles on one.

    The counters are the diagnostic: ``.graphs`` recordings kept,
    ``.replays`` calls served by one, ``.misses`` calls that ran eagerly. A
    step that never replays is one whose shapes are not settling.
    """
    if fn is None:
        return lambda f: graphed(f, max_graphs=max_graphs, enabled=enabled)
    if not enabled:
        return fn

    captured: collections.OrderedDict = collections.OrderedDict()
    seen: collections.OrderedDict = collections.OrderedDict()

    @functools.wraps(fn)
    def run(*args, **kwargs):
        key = (_signature(args), _signature(kwargs))
        held = captured.get(key)
        if held is not None:
            captured.move_to_end(key)
            fresh: list = []
            _tensors((args, kwargs), fresh)
            run.replays += 1
            return held.replay(fresh)

        if key not in seen:
            # first sighting: run it, and remember the shape. A step whose
            # shapes never repeat never pays for a recording.
            seen[key] = True
            while len(seen) > 4 * max_graphs:
                seen.popitem(last=False)
            run.misses += 1
            return fn(*args, **kwargs)

        fresh = []
        _tensors((args, kwargs), fresh)
        # the tensors this call reads must outlive it: the graph names their
        # buffers, and the caller is free to drop its own
        kept = [t.clone() for t in fresh]
        kept_args, kept_kwargs = _rebuild((args, kwargs), iter(kept))
        _C.begin_capture()
        try:
            outputs = fn(*kept_args, **kept_kwargs)
        except BaseException:
            _C.end_capture()  # never leave a capture open on the device
            raise
        captured[key] = _Graphed(kept, outputs, _C.end_capture())
        while len(captured) > max_graphs:
            captured.popitem(last=False)
        run.graphs, run.captures = len(captured), run.captures + 1
        return outputs

    run.graphs = run.captures = run.replays = run.misses = 0
    run.captured = captured
    return run
