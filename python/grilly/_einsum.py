"""torch.einsum as permutes, sums and one batched matmul per pair of
operands, so it is differentiable through ops that already are.

For two operands, every label is one of:
  batch       in both inputs and the output -> the bmm batch
  contracted  in both inputs, not the output -> the bmm K
  kept        in one input and the output    -> that side's M or N
  summed      in one input only, not output  -> summed out first
More operands are contracted left to right, keeping the labels a later
operand or the output still needs. Repeated labels within one operand
(diagonals, "ii->i") are not implemented yet.
"""

from __future__ import annotations

import builtins
import math

from . import Tensor, _not_yet, _require_tensor

_ELLIPSIS_LABELS = "\x00\x01\x02\x03\x04\x05\x06\x07"  # stand-ins for "..." dims, never user labels


def _err(msg: str) -> RuntimeError:
    return RuntimeError(f"einsum(): {msg}")


def _parse(equation: str, operands):
    eq = equation.replace(" ", "")
    lhs, arrow, rhs = eq.partition("->")
    terms = lhs.split(",")
    if len(terms) > len(operands):
        raise _err("fewer operands were provided than specified in the equation")
    if len(terms) < len(operands):
        raise _err("more operands were provided than specified in the equation")
    ell_rank = 0
    subs = []
    for n, (term, x) in enumerate(zip(terms, operands)):
        if "..." in term:
            head, _, tail = term.partition("...")
            if "." in head + tail:
                raise _err(f"found '.' for operand {n} that is not part of any ellipsis")
            rank = x.dim() - len(head) - len(tail)
            if rank < 0:
                raise _err(f"the number of subscripts in the equation ({len(head) + len(tail)}) is more than the "
                           f"number of dimensions ({x.dim()}) for operand {n}")
            ell_rank = builtins.max(ell_rank, rank)
            # right-aligned, so operands with fewer "..." dims broadcast
            labels = list(head) + list(_ELLIPSIS_LABELS[len(_ELLIPSIS_LABELS) - rank:]) + list(tail)
        else:
            if len(term) != x.dim():
                raise _err(f"the number of subscripts in the equation ({len(term)}) does not match the number of "
                           f"dimensions ({x.dim()}) for operand {n} and no ellipsis was given")
            labels = list(term)
        for c in labels:
            if c not in _ELLIPSIS_LABELS and not c.isalpha():
                raise _err(f"invalid subscript given at index {eq.index(c)} in the equation string, subscripts "
                           "must be in [a-zA-Z]")
        subs.append(labels)
    ell = list(_ELLIPSIS_LABELS[len(_ELLIPSIS_LABELS) - ell_rank:]) if ell_rank else []
    if arrow:
        out = []
        if "..." in rhs:
            head, _, tail = rhs.partition("...")
            out = list(head) + ell + list(tail)
        else:
            out = list(rhs)
        seen = set()
        for c in out:
            if c in seen:
                raise _err(f"output subscript {c} appears more than once in the output")
            seen.add(c)
            if not builtins.any(c in s for s in subs):
                raise _err(f"output subscript {c} does not appear in the equation for any input operand")
    else:
        counts = {}
        for s in subs:
            for c in s:
                counts[c] = counts.get(c, 0) + 1
        out = ell + sorted(c for c, k in counts.items() if k == 1 and c not in _ELLIPSIS_LABELS)
    return subs, out


def _sizes(subs, operands):
    """Each label's size; a size of 1 broadcasts, as in torch."""
    size = {}
    for n, (labels, x) in enumerate(zip(subs, operands)):
        if len(set(labels)) != len(labels):
            raise _not_yet("einsum with a label repeated within one operand (a diagonal)")
        for c, s in zip(labels, x.shape):
            prev = size.get(c)
            if prev is None or prev == 1:
                size[c] = s
            elif s != prev and s != 1:
                name = "..." if c in _ELLIPSIS_LABELS else c
                raise _err(f"subscript {name} has size {s} for operand {n} which does not broadcast with "
                           f"previously seen size {prev}")
    return size


def _sum_out(x: Tensor, labels, keep):
    """Sum the dims whose labels are not in `keep`."""
    drop = [d for d, c in enumerate(labels) if c not in keep]
    if drop:
        x = x.sum(drop)
        labels = [c for c in labels if c in keep]
    return x, labels


def _arrange(x: Tensor, labels, order, size):
    """Permute `x` to `order` (labels it has), expanding size-1 dims."""
    x = x.permute(*[labels.index(c) for c in order]) if order != labels else x
    want = [size[c] for c in order]
    if list(x.shape) != want:
        x = x.expand(*want)
    return x


def _pair(a: Tensor, la, b: Tensor, lb, needed, size):
    """Contract two operands, keeping the labels in `needed`."""
    a, la = _sum_out(a, la, set(lb) | needed)
    b, lb = _sum_out(b, lb, set(la) | needed)
    batch = [c for c in la if c in lb and c in needed]
    contracted = [c for c in la if c in lb and c not in needed]
    a_only = [c for c in la if c not in lb]
    b_only = [c for c in lb if c not in la]
    nb = math.prod(size[c] for c in batch)
    m = math.prod(size[c] for c in a_only)
    k = math.prod(size[c] for c in contracted)
    n = math.prod(size[c] for c in b_only)
    a3 = _arrange(a, la, batch + a_only + contracted, size).reshape(nb, m, k)
    b3 = _arrange(b, lb, batch + contracted + b_only, size).reshape(nb, k, n)
    r = (a3 @ b3).reshape(*[size[c] for c in batch + a_only + b_only])
    return r, batch + a_only + b_only


def einsum(equation, *operands) -> Tensor:
    """torch.einsum(equation, *operands) (operands may also be one list)."""
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    ops = [_require_tensor(f"operand {i}", x) for i, x in enumerate(operands)]
    subs, out = _parse(equation, ops)
    size = _sizes(subs, ops)
    x, labels = ops[0], subs[0]
    for i in range(1, len(ops)):
        needed = set(out).union(*[set(s) for s in subs[i + 1:]])
        x, labels = _pair(x, labels, ops[i], subs[i], needed, size)
    x, labels = _sum_out(x, labels, set(out))
    return _arrange(x, labels, out, size)
