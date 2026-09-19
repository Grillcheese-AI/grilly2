"""``AutoTokenizer``: Hugging Face's ``tokenizers`` plus the parts of
``transformers``' tokenizer that a chat prompt needs.

grilly does not implement a tokenizer (CLAUDE.md §11). This wraps
``tokenizers.Tokenizer`` — the Rust one, which is what transformers uses
underneath for a fast tokenizer — and reads ``tokenizer_config.json`` for
the special tokens and the chat template. Rendering a template needs
``jinja2``, as it does in transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import Tensor, _not_yet, tensor
from .loading import repo_files

__all__ = ["AutoTokenizer", "Tokenizer"]

FILES = ("tokenizer.json", "tokenizer_config.json")


class Tokenizer:
    """What ``AutoTokenizer.from_pretrained`` returns: ``encode``, ``decode``,
    ``apply_chat_template`` and the special token ids."""

    def __init__(self, backend, config: dict) -> None:
        self._t = backend
        self.config = config
        self.chat_template = config.get("chat_template")
        self.eos_token = self._token_str(config.get("eos_token"))
        self.bos_token = self._token_str(config.get("bos_token"))
        self.pad_token = self._token_str(config.get("pad_token")) or self.eos_token

    @staticmethod
    def _token_str(value):
        if isinstance(value, dict):  # {"content": "<|im_end|>", ...}
            return value.get("content")
        return value

    def _id(self, token):
        return None if token is None else self._t.token_to_id(token)

    @property
    def eos_token_id(self):
        return self._id(self.eos_token)

    @property
    def bos_token_id(self):
        return self._id(self.bos_token)

    @property
    def pad_token_id(self):
        return self._id(self.pad_token)

    @property
    def vocab_size(self) -> int:
        return self._t.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._t.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, Tensor):
            ids = ids.numpy().reshape(-1).tolist()
        return self._t.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        if isinstance(sequences, Tensor):
            sequences = sequences.numpy().tolist()
        return [self.decode(s, skip_special_tokens) for s in sequences]

    def __call__(self, text, return_tensors=None, add_special_tokens: bool = True) -> dict:
        """``tokenizer(text, return_tensors="pt")``: ``input_ids`` and an
        ``attention_mask`` of ones, batched. Padding is not implemented, so
        a list of texts must be of equal length once encoded."""
        texts = [text] if isinstance(text, str) else list(text)
        rows = [self.encode(t, add_special_tokens) for t in texts]
        if len({len(r) for r in rows}) > 1:
            raise _not_yet("tokenizing texts of different lengths together (padding)")
        return {"input_ids": self._wrap(rows, return_tensors),
                "attention_mask": self._wrap([[1] * len(r) for r in rows], return_tensors)}

    @staticmethod
    def _wrap(rows, return_tensors):
        if return_tensors in (None, "list"):
            return rows
        if return_tensors != "pt":
            raise _not_yet(f"return_tensors={return_tensors!r} (grilly returns grilly tensors; use 'pt')")
        return tensor(rows)

    def apply_chat_template(self, conversation, add_generation_prompt: bool = False, tokenize: bool = True,
                            return_tensors=None, **template_kwargs):
        """Render the model's chat template, as transformers does. Without
        ``tokenize`` the rendered string comes back instead of ids."""
        if self.chat_template is None:
            raise ValueError("this tokenizer has no chat_template in tokenizer_config.json; "
                             "build the prompt yourself and call encode()")
        try:
            from jinja2 import Environment
            from jinja2.exceptions import TemplateError
        except ImportError:  # pragma: no cover - depends on the environment
            raise _not_yet("apply_chat_template without jinja2 installed "
                           "(pip install jinja2, as transformers needs it too)") from None

        def raise_exception(message):
            raise TemplateError(message)

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        rendered = env.from_string(self.chat_template).render(
            messages=list(conversation), add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "", eos_token=self.eos_token or "", **template_kwargs)
        if not tokenize:
            return rendered
        ids = self.encode(rendered, add_special_tokens=False)
        return self._wrap([ids], return_tensors) if return_tensors else ids


class AutoTokenizer:
    """``transformers.AutoTokenizer`` for repositories with a
    ``tokenizer.json`` (every model converted for the fast tokenizer)."""

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs) -> Tokenizer:
        try:
            from tokenizers import Tokenizer as Backend
        except ImportError:  # pragma: no cover - depends on the environment
            raise _not_yet("AutoTokenizer without Hugging Face tokenizers installed "
                           "(pip install tokenizers; grilly does not implement a tokenizer)") from None
        files = repo_files(pretrained_model_name_or_path, FILES, optional=("tokenizer_config.json",), **kwargs)
        config = {}
        if "tokenizer_config.json" in files:
            config = json.loads(Path(files["tokenizer_config.json"]).read_text(encoding="utf-8"))
        return Tokenizer(Backend.from_file(files["tokenizer.json"]), config)
