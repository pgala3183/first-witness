"""A Fireworks training-pool model's chat template + tokenizer, without transformers.

The fine-tuned call agent has no chat endpoint (that would need a dedicated
deployment), so prompts are rendered client-side and sent as token ids. The
training script (finetune/serverless.py) and the live call (call_model.py)
both use this class, so the model is called exactly the way it was trained.

Chat.prompt_ids(messages) is the generation prompt with thinking off; the
training target is the answer followed by <|im_end|>, so prompt + target is
exactly the template's rendering of the full conversation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

END = "<|im_end|>"
# Fireworks base model -> Hugging Face repo with the matching tokenizer + chat template.
TOKENIZERS = {
    "accounts/fireworks/models/qwen3p6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "accounts/fireworks/models/qwen3p8-27b": "Qwen/Qwen3.8-27B",
}


class Chat:
    def __init__(self, repo: str) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from huggingface_hub import hf_hub_download
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        try:
            template = Path(hf_hub_download(repo, "chat_template.jinja")).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001  (older repos keep it in tokenizer_config.json)
            cfg = json.loads(Path(hf_hub_download(repo, "tokenizer_config.json")).read_text(encoding="utf-8"))
            template = cfg["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: (
            json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys))

        def _raise(msg: str) -> None:
            raise ValueError(msg)

        env.globals["raise_exception"] = _raise
        self.template = env.from_string(template)
        self.end_id = self.tok.token_to_id(END)
        if self.end_id is None:
            raise ValueError(f"{repo}: no {END} token")

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        return self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        return self.tok.encode(self.prompt_text(messages), add_special_tokens=False).ids

    def target_ids(self, answer: str) -> list[int]:
        return self.tok.encode(answer + END, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        ids = list(ids)
        if self.end_id in ids:
            ids = ids[: ids.index(self.end_id)]
        return self.tok.decode(ids, skip_special_tokens=True).strip()


@lru_cache(maxsize=4)
def chat_for(base_model: str) -> Chat:
    repo = TOKENIZERS.get(base_model)
    if not repo:
        raise KeyError(f"No tokenizer mapping for {base_model}; add it to TOKENIZERS in {Path(__file__).name}.")
    return Chat(repo)


def connect(api_key: str):
    """One client for training and sampling on Fireworks' shared serverless pool."""
    from fireworks.training.sdk import FiretitanServiceClient

    root = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/")
    return FiretitanServiceClient(api_key=api_key, base_url=f"{root}/training/v1/serverless")
