# `grilly.huggingface`: the adapter

```python
from grilly.huggingface import AutoModelForCausalLM, AutoTokenizer
from grilly.infer.quantization import Int8WeightOnlyConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             quantization_config=Int8WeightOnlyConfig())
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tokenizer.apply_chat_template([{"role": "user", "content": "Why is the sky blue?"}],
                                    add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(model.generate(ids, max_new_tokens=128)[0]))
```

That is `transformers`' own code with one import changed, and it runs on a
consumer AMD card under Windows.

## What it is, and what it is not

CLAUDE.md §11 draws the line: **the machinery** — models, safetensors
loading, quantization, the KV cache, samplers, `generate` — is
`grilly.infer`, and **the adapter** is this package: `config.json` parsing,
file resolution, `from_pretrained`, a tokenizer wrapper. The adapter must
be deletable.

`tests/python/test_huggingface.py::test_the_machinery_works_without_the_adapter`
is that line as a test. It starts a fresh interpreter, builds a model from
`grilly.infer.models`, loads a safetensors checkpoint, quantizes it,
generates, and then asserts `"grilly.huggingface" not in sys.modules`. If
someone routes a mechanism through the adapter, that test fails.

| in the adapter | in the machinery |
|---|---|
| `config.json` → `LlamaConfig`, architecture → model class | the model itself |
| finding files: a directory, or the Hub cache through `huggingface_hub` | `safetensors.stream_into`, which reads one tensor at a time |
| `from_pretrained` kwargs, and refusing the ones grilly cannot honour | `load_quantized_`, `quantize_` |
| `generation_config.json` → `GenerationConfig` | `generate`, the KV cache, the samplers |
| `AutoTokenizer`: Hugging Face `tokenizers` plus the chat template | — (grilly does not implement a tokenizer) |

## Configs

`configuration.py` is the only file that knows Hugging Face's field names.
Llama and Qwen2 are one decoder in grilly; they differ in where biases go,
and that is the one subtlety worth stating: **Qwen2 biases q, k and v and
never the output projection, while Llama's `attention_bias` covers all
four.** `tests/python/test_huggingface.py` checks every other field against
`transformers.AutoConfig` on the same file, so the mapping is not taken on
trust.

Two versions of the same field exist in the wild: transformers 5 moved
`rope_theta` into `rope_parameters` beside a `rope_type`. Both are read,
and a `rope_type` other than `default` is **refused**, not ignored —
silently falling back to a 10,000 theta would produce a model that loads,
runs, and is wrong.

Everything grilly's decoder does not implement is refused by name:
`rope_scaling`, sliding-window attention, mixture-of-experts fields,
non-zero attention dropout, an unknown `model_type`, a non-causal
architecture, `trust_remote_code=True`, a dtype other than float32, a
`device_map` naming more than one device, and any `generation_config.json`
field that would change what is generated (`no_repeat_ngram_size`, beam
settings, and the rest). Arguments that genuinely make no difference here —
`low_cpu_mem_usage`, `use_safetensors`, `attn_implementation` — are
accepted and ignored, with the reason written next to each.

## Loading

Sharded checkpoints load from `model.safetensors.index.json`, shard by
shard. With a `quantization_config`, each weight is quantized as it is read
(`docs/inference.md` §7), so a checkpoint never exists in float32 in full.
A tied output projection is restored after loading and quantized then; a
weight tied to one the checkpoint carried is not reported missing.

Two traps found while building this, both now tests:

- **holding a `state_dict` across a streaming load** keeps every weight the
  loop has already replaced alive, which is exactly the memory the
  streaming exists to avoid. `stream_into` takes shapes and nothing else.
  `test_load_quantized.py` measures the peak and fails with "the load is
  holding on to weights it has replaced".
- **`state_dict()` on a quantized model dequantizes every weight at once** —
  the whole model in float32, and on this card it stayed in the pool
  afterwards: 1.24 GiB of dedicated VRAM while generating became 3.13.
  Loading code asks for `keep_vars=True`. Dequantizing is a kernel, so the
  test asserts on the dispatch counter.

## The tokenizer

`AutoTokenizer.from_pretrained` returns a wrapper over
`tokenizers.Tokenizer` — the Rust tokenizer transformers itself uses —
with `encode`, `decode`, `batch_decode`, `__call__(..., return_tensors="pt")`,
the special-token ids, and `apply_chat_template`. Templates are Jinja, as
in transformers, so rendering needs `jinja2`; the test renders one against
`transformers.AutoTokenizer` and requires the same string out.

Padding is not implemented: tokenizing several texts of different lengths
together raises rather than silently truncating or padding.

## The acceptance run

`python tools/generate.py` is this package end to end — load, quantize
while loading, chat template, sample, report. On an RX 6750 XT with
Qwen2.5-0.5B-Instruct: **166 tokens/s**, 1.24 GiB while generating, 2.4 GiB
peak while loading, and greedy float32 generation identical to transformers
for 32 tokens with and without a repetition penalty.

## Known gap: constructing a model still materializes weights

`from_pretrained` builds the model and then overwrites it. Building
Qwen2.5-0.5B allocates and initializes 1.84 GiB of float32 parameters the
checkpoint is about to replace, plus 0.5 GiB for the untied `lm_head`
weight that `tie_weights` immediately frees. Streaming cannot help: both
happen before it runs.

For a 0.5B model on 12 GB this is merely wasteful. For a 7B model it is
fatal — the int8 weights would fit in about 7 GB, but constructing the
float32 model first needs 28 GB. **So this is what decides how large a
model this adapter can load**, and it is an open decision (torch spells the
fix `with torch.device("meta")`, accelerate `init_empty_weights`).
