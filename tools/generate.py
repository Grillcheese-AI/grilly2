"""The CLAUDE.md §11 acceptance run: load a small open-weights model from
safetensors, quantize it, generate sampled text on the GPU, and report
tokens/sec and peak VRAM.

    python tools/generate.py                          # Qwen2.5-0.5B-Instruct from the HF cache
    python tools/generate.py --check-hf               # first: greedy float32 tokens vs transformers
    python tools/generate.py --model DIR --prompt "..."

The model directory holds `config.json`, `generation_config.json`,
`model.safetensors` and `tokenizer.json`. Without --model it is looked up in
the local Hugging Face cache (huggingface_hub), and nothing is downloaded.
Reading config.json into a LlamaConfig here is what grilly-huggingface's
adapter will do; grilly-infer itself takes no Hugging Face files. The
tokenizer is Hugging Face's `tokenizers`, per CLAUDE.md §11.

Qwen2 only so far: Llama-family configs load too, but the prompt uses
Qwen's chat format.
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
from grilly.infer.models import LlamaConfig, LlamaForCausalLM
from grilly.infer.quantization import Int8WeightOnlyConfig, quantize_
from grilly.infer.safetensors import load_file

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")


def model_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from huggingface_hub import try_to_load_from_cache
    paths = [try_to_load_from_cache(REPO, f) for f in FILES]
    missing = [f for f, p in zip(FILES, paths) if not isinstance(p, str)]
    if missing:
        sys.exit(f"{REPO} is not fully in the local Hugging Face cache (missing {missing}); pass --model DIR")
    return Path(paths[0]).parent


def config_from(path: Path) -> LlamaConfig:
    c = json.loads((path / "config.json").read_text())
    if c.get("model_type") not in ("qwen2", "llama"):
        sys.exit(f"model_type {c.get('model_type')!r}: this tool knows qwen2 and llama")
    if c.get("rope_scaling"):
        sys.exit(f"rope_scaling {c['rope_scaling']} is not supported yet")
    if c.get("use_sliding_window"):
        sys.exit("sliding-window attention is not supported yet")
    qwen = c["model_type"] == "qwen2"
    return LlamaConfig(
        vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c.get("num_key_value_heads"), head_dim=c.get("head_dim"),
        rms_norm_eps=c["rms_norm_eps"], rope_theta=c.get("rope_theta", 10000.0),
        qkv_bias=qwen or c.get("attention_bias", False), o_bias=(not qwen) and c.get("attention_bias", False),
        tie_word_embeddings=c.get("tie_word_embeddings", False))


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
    text = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer.encode(text).ids


def check_against_transformers(path: Path, model: LlamaForCausalLM, ids: list[int], n: int, penalty: float) -> None:
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
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-hf", action="store_true", help="compare greedy float32 tokens with transformers first")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    path = model_dir(args.model)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    gen = json.loads((path / "generation_config.json").read_text())
    config = config_from(path)
    ids = chat(tokenizer, args.prompt)

    with PeakVram() as load_vram:
        t0 = time.perf_counter()
        model = LlamaForCausalLM(config)
        model.load_state_dict(load_file(path / "model.safetensors"), assign=True, strict=False)
        model.tie_weights()
        grilly._C.synchronize()
        print(f"loaded {config.num_hidden_layers} layers, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters "
              f"in {time.perf_counter() - t0:.1f} s")
        if args.check_hf:
            for penalty in (1.0, gen.get("repetition_penalty", 1.0)):
                check_against_transformers(path, model, ids, 32, penalty)
        if not args.no_quantize:
            t0 = time.perf_counter()
            quantize_(model, Int8WeightOnlyConfig())
            grilly._C.synchronize()
            print(f"quantized to int8 in {time.perf_counter() - t0:.2f} s")

        settings = {k: gen[k] for k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty") if k in gen}
        eos = gen.get("eos_token_id")
        print(f"sampling with the model's generation_config: {settings}, eos {eos}")

        class Stream:
            """Prints text as it comes, and times the first and last token."""
            def __init__(self):
                self.tokens, self.times = [], []

            def put(self, value):
                if not self.times:  # the prompt
                    self.times.append(time.perf_counter())
                    return
                self.tokens.extend(value.tolist())
                self.times.append(time.perf_counter())
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
    print(f"GPU work: {(after['dispatches'] - before['dispatches']) / n:.0f} dispatches and "
          f"{(after['wait_ns_total'] - before['wait_ns_total']) / n / 1e6:.1f} ms of GPU wait per token, "
          f"{(after['submits'] - before['submits']) / n:.1f} submits per token")
    print(f"weights and cache in persistent memory: {after['persistent_bytes'] / 2**30:.2f} GiB")
    if vram.peak is not None:
        print(f"peak dedicated VRAM: {load_vram.peak / 2**30:.2f} GiB while loading and quantizing (float32 weights first), "
              f"{vram.peak / 2**30:.2f} GiB while generating")


if __name__ == "__main__":
    main()
