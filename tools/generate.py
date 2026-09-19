"""The CLAUDE.md §11 acceptance run: load a small open-weights model from
safetensors, quantize it, generate sampled text on the GPU, and report
tokens/sec and peak VRAM.

    python tools/generate.py                          # Qwen2.5-0.5B-Instruct from the HF cache
    python tools/generate.py --check-hf               # first: greedy float32 tokens vs transformers
    python tools/generate.py --model DIR --prompt "..."

The whole run is `grilly.huggingface`, which is `transformers`' own calls:

    model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=Int8WeightOnlyConfig())
    tokenizer = AutoTokenizer.from_pretrained(repo)
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    out = model.generate(ids, **settings)

Nothing is downloaded: the repository must already be in the local Hugging
Face cache, or --model must name a directory holding `config.json`,
`generation_config.json`, `model.safetensors` and `tokenizer.json`. The
machinery underneath does not need the adapter (CLAUDE.md §11); what that
looks like without it is `tests/python/test_huggingface.py`'s last test.

Llama and Qwen2 configs load; the chat prompt is whatever the tokenizer's
own template renders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import grilly
from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer
from grilly.infer.quantization import Int8WeightOnlyConfig

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")


def model_dir(arg: str | None) -> Path:
    """The directory the files are in: --model, or the local Hugging Face
    cache. Nothing is downloaded."""
    if arg:
        return Path(arg)
    from huggingface_hub import try_to_load_from_cache
    paths = [try_to_load_from_cache(REPO, f) for f in FILES]
    missing = [f for f, p in zip(FILES, paths) if not isinstance(p, str)]
    if missing:
        sys.exit(f"{REPO} is not fully in the local Hugging Face cache (missing {missing}); pass --model DIR")
    return Path(paths[0]).parent


class PeakVram:
    """This process's peak dedicated GPU memory, from Windows' GPU counters
    (sampled about twice a second). Elsewhere it reports nothing."""

    def __init__(self) -> None:
        self.peak = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True) if os.name == "nt" else None

    def _sample(self):
        q = (f"$c = Get-Counter -ErrorAction SilentlyContinue '\\GPU Process Memory(pid_{os.getpid()}_*)\\Dedicated Usage';"
             "($c.CounterSamples | Measure-Object CookedValue -Sum).Sum")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True).stdout.strip()
        return float(out) if out else None

    def _run(self):
        while not self._stop.is_set():
            v = self._sample()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)

    def __enter__(self):
        if self._thread:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()


def chat(tokenizer, prompt: str) -> list[int]:
    """The prompt through the model's own chat template."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)


def check_against_transformers(path: Path, model, ids: list[int], n: int, penalty: float) -> None:
    """Greedy float32 tokens from both, with the same settings passed to
    both explicitly: from_pretrained also loads generation_config.json, whose
    repetition_penalty transformers would otherwise apply to its greedy run."""
    import torch
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    kw = dict(max_new_tokens=n, do_sample=False, repetition_penalty=penalty)
    with torch.no_grad():
        want = ref.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                            top_k=None, top_p=None, temperature=None, **kw).tolist()[0]
    got = model.generate(grilly.tensor([ids]), **kw).tolist()[0]
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    same = first is None and len(got) == len(want)
    print(f"greedy float32, repetition_penalty {penalty}, {n} new tokens: "
          f"{'identical to transformers' if same else f'first difference at new token {first - len(ids)}'}")
    del ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model directory (default: Qwen2.5-0.5B-Instruct from the HF cache)")
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--repeat-prompt", type=int, default=1, help="repeat the prompt N times, to decode against a long cache")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-hf", action="store_true", help="compare greedy float32 tokens with transformers first")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    args.prompt *= args.repeat_prompt

    path = model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = chat(tokenizer, args.prompt)

    grilly._C.reset_persistent_peak()
    with PeakVram() as load_vram:
        t0 = time.perf_counter()
        # float32 first when comparing with transformers; otherwise each weight
        # is quantized as it is read, and no float32 copy of the model exists
        quantization = None if args.no_quantize or args.check_hf else Int8WeightOnlyConfig()
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quantization)
        grilly._C.synchronize()
        print(f"loaded {model.config.num_hidden_layers} layers, "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters "
              f"in {time.perf_counter() - t0:.1f} s{'' if quantization is None else ' (quantized while loading)'}")
        gen = vars(model.generation_config)
        if args.check_hf:
            for penalty in (1.0, gen.get("repetition_penalty", 1.0)):
                check_against_transformers(path, model, ids, 32, penalty)
        if quantization is None and not args.no_quantize:
            from grilly.infer.quantization import quantize_
            t0 = time.perf_counter()
            quantize_(model, Int8WeightOnlyConfig())
            grilly._C.synchronize()
            print(f"quantized to int8 in {time.perf_counter() - t0:.2f} s")
        load_peak = grilly._C.submit_stats()["persistent_peak"]

        settings = {k: gen[k] for k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")
                    if gen.get(k) is not None}
        eos = gen.get("eos_token_id")
        print(f"sampling with the model's generation_config: {settings}, eos {eos}")

        class Stream:
            """Prints text as it comes, and times the first and last token."""
            def __init__(self):
                self.tokens, self.times, self.mid = [], [], None

            def put(self, value):
                if not self.times:  # the prompt
                    self.times.append(time.perf_counter())
                    return
                self.tokens.extend(value.tolist())
                self.times.append(time.perf_counter())
                if self.mid is None:  # prefill is over: count the decode steps alone
                    self.mid = grilly._C.submit_stats()
                print(tokenizer.decode(self.tokens[-1:]), end="", flush=True)

            def end(self):
                print()

    grilly.manual_seed(args.seed)
    stream = Stream()
    grilly._C.synchronize()
    before = grilly._C.submit_stats()
    with PeakVram() as vram:
        start = time.perf_counter()
        model.generate(grilly.tensor([ids]), max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                       streamer=stream, **settings)
    after = grilly._C.submit_stats()
    n = len(stream.tokens)
    prefill = stream.times[1] - start
    decode = stream.times[-1] - stream.times[1]
    print(f"\nprompt {len(ids)} tokens, prefill + first token {prefill * 1e3:.0f} ms; "
          f"{n} new tokens, decode {(n - 1) / decode:.1f} tokens/s ({decode / max(n - 1, 1) * 1e3:.1f} ms/token)")
    steps = max(n - 1, 1)
    mid = stream.mid or before
    print(f"prefill: {mid['dispatches'] - before['dispatches']} dispatches, "
          f"{(mid['wait_ns_total'] - before['wait_ns_total']) / 1e6:.0f} ms of GPU wait")
    print(f"per decoded token: {(after['dispatches'] - mid['dispatches']) / steps:.0f} dispatches and "
          f"{(after['wait_ns_total'] - mid['wait_ns_total']) / steps / 1e6:.1f} ms of GPU wait, "
          f"{(after['submits'] - mid['submits']) / steps:.1f} submits")
    print(f"weights and cache in persistent memory: {after['persistent_bytes'] / 2**30:.2f} GiB, "
          f"peak while loading {load_peak / 2**30:.2f} GiB")
    if vram.peak is not None:
        print(f"peak dedicated VRAM: {load_vram.peak / 2**30:.2f} GiB while loading and quantizing (float32 weights first), "
              f"{vram.peak / 2**30:.2f} GiB while generating")


if __name__ == "__main__":
    main()
