"""int4 with GPTQ error compensation, against int4 round-to-nearest.

`tools/int4_accuracy.py` bounds what rounding each weight on its own can
ever do: about 1.18x better than grilly's round-to-nearest, and still nine
times worse than int8. GPTQ leaves that family — it adjusts the columns it
has not quantized yet so the error cancels in the layer's *output*, in the
metric of that layer's real inputs (`grilly.infer.gptq`).

This runs it on a real model and measures whether it pays:

    python tools/gptq.py                       # Qwen2.5-0.5B-Instruct from the cache
    python tools/gptq.py --samples 32 --length 512

It reports, for the same prompt: the layer-by-layer output error against
the float32 model, the perplexity of a held-out slice of wikitext, and what
each scheme generates. Calibration text is wikitext-2, the corpus GPTQ's
own experiments use (about 5 MB, downloaded once).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import grilly
from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer
from grilly.infer.calibration import calibration_ids, quantize_gptq_, wikitext_texts
from grilly.infer.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

sys.path.insert(0, str(Path(__file__).parent))
from generate import model_dir  # noqa: E402


def perplexity(model, ids: grilly.Tensor, stride: int = 512) -> float:
    """Token-level perplexity of `ids` (1, n), in float64 on the host."""
    total, count = 0.0, 0
    with grilly.no_grad():
        for start in range(0, ids.shape[1] - 1, stride):
            window = ids[:, start:start + stride + 1]
            if window.shape[1] < 2:
                break
            logits = model(window[:, :-1]).logits.numpy().astype(np.float64)[0]
            targets = window.numpy()[0, 1:]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            total += -log_probs[np.arange(len(targets)), targets].sum()
            count += len(targets)
    return float(np.exp(total / count))


def generate(model, tokenizer, prompt: str, tokens: int = 40) -> str:
    ids = grilly.tensor([tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True)])
    grilly.manual_seed(0)
    out = model.generate(ids, max_new_tokens=tokens, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model directory (default: Qwen2.5-0.5B-Instruct from the HF cache)")
    ap.add_argument("--samples", type=int, default=16, help="calibration sequences")
    ap.add_argument("--length", type=int, default=512, help="tokens per calibration sequence")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    args = ap.parse_args()

    path = model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"calibration: {args.samples} sequences of {args.length} tokens from wikitext-2")
    texts = wikitext_texts(samples=400)
    half = len(texts) // 2
    ids = calibration_ids(tokenizer, texts[:half], samples=args.samples, length=args.length)
    # held out: text the calibration never saw, as one long sequence
    window = calibration_ids(tokenizer, texts[half:], samples=4, length=args.length, seed=1)
    held_out = grilly.tensor([[int(v) for row in window.numpy() for v in row]])
    print(f"  {tuple(ids.shape)} calibration, {tuple(held_out.shape)} held out for perplexity\n")

    def fresh():
        return AutoModelForCausalLM.from_pretrained(path)

    results = {}
    reference = fresh()
    results["float32"] = (perplexity(reference, held_out), generate(reference, tokenizer, args.prompt))
    probe = grilly.tensor([tokenizer.encode("The sky is blue because")])
    with grilly.no_grad():
        want = reference(probe).logits.numpy()
    del reference

    for name, prepare in (
        ("int8 (per row)", lambda m: quantize_(m, Int8WeightOnlyConfig())),
        ("int4 (round to nearest)", lambda m: quantize_(m, Int4WeightOnlyConfig(group_size=args.group_size))),
        ("int4 (GPTQ)", "gptq"),
        ("int4 (GPTQ, act-order)", "act"),
    ):
        model = fresh()
        started = time.perf_counter()
        if callable(prepare):
            prepare(model)
        else:
            quantize_gptq_(model, ids, group_size=args.group_size,
                           act_order=(prepare == "act"), report=None)
            # the output projection is left to round-to-nearest, as AutoGPTQ
            # and torchao do: 152K rows, and the least sensitive layer
            quantize_(model, Int4WeightOnlyConfig(group_size=args.group_size))
        elapsed = time.perf_counter() - started
        with grilly.no_grad():
            got = model(probe).logits.numpy()
        logit_error = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        results[name] = (perplexity(model, held_out), generate(model, tokenizer, args.prompt))
        print(f"{name}: quantized in {elapsed:.1f} s, logit error {logit_error:.3e}")
        del model

    print(f"\n{'scheme':26} {'perplexity':>11}")
    for name, (ppl, _) in results.items():
        print(f"{name:26} {ppl:11.3f}")
    print()
    for name, (_, text) in results.items():
        print(f"{name}:\n  {text.strip()[:300]}\n")


if __name__ == "__main__":
    main()
