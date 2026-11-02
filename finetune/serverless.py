"""Fine-tune the call agent on Fireworks serverless training (shared GPUs).

No deployment and no trainer job: a pooled LoRA trainer runs our training
loop and serves the adapter for sampling in the same session, billed per
token. That is what lets us train and evaluate without a dedicated GPU.

  python finetune/serverless.py check                      # 3 answers from the base model
  python finetune/serverless.py train --limit-train 64     # tiny paid smoke run (cents)
  python finetune/serverless.py train                      # full run: 3,300 rows x 2 epochs
  python finetune/serverless.py promote --as first-witness-call-q36   # register the result as a model

Evaluate base vs fine-tuned with the usual grader:
  python finetune/eval.py --system serverless --model accounts/fireworks/models/qwen3p6-35b-a3b \\
      --split test_real --split test_synth --name sl-base-q36
  python finetune/eval.py --system serverless --snapshot latest \\
      --split test_real --split test_synth --name sl-ft-q36

Prompts are rendered client-side with the model's own chat template (thinking
off), exactly as the live call would send them; the loss covers only the
answer tokens. Needs FIREWORKS_API_KEY and `pip install "fireworks-ai[training]"`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
import envfile  # noqa: E402,F401  (FIREWORKS_API_KEY from the repo .env)

sys.path.insert(0, str(HERE.parent))
# The live call renders prompts with the same code: train on exactly what it sends.
from services.voice.chat_template import END, TOKENIZERS, Chat  # noqa: E402,F401
from services.voice.chat_template import chat_for as _chat_for  # noqa: E402
from services.voice.chat_template import connect as _connect  # noqa: E402
DATA = HERE / "data"
RESULTS = HERE / "results"
RUN_FILE = RESULTS / "serverless_run.json"

BASE_MODEL = "accounts/fireworks/models/qwen3p6-35b-a3b"
# USD per 1M tokens, Fireworks serverless training page (Nov 2026). Prefill, sample, train.
PRICES = {
    "accounts/fireworks/models/qwen3p6-35b-a3b": (0.45, 1.335, 2.14),
    "accounts/fireworks/models/qwen3p8-27b": (1.86, 5.595, 4.103),
}
MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------- chat template

def chat_for(base_model: str) -> Chat:
    try:
        return _chat_for(base_model)
    except KeyError as e:
        sys.exit(str(e))


def sft_tokens(chat: Chat, messages: list[dict[str, str]]) -> tuple[list[int], list[float]] | None:
    """(tokens, weights) for [system, user, assistant]: weight 1 on the answer only."""
    prompt = chat.prompt_ids(messages[:2])
    target = chat.target_ids(messages[2]["content"])
    tokens = (prompt + target)[:MAX_SEQ_LEN]
    weights = ([0.0] * len(prompt) + [1.0] * len(target))[:MAX_SEQ_LEN]
    return (tokens, weights) if any(weights[1:]) else None


def to_datum(tokens: list[int], weights: list[float]):
    """Next-token layout: the model reads tokens[:-1] and is scored on tokens[1:]."""
    from fireworks.training import sdk

    return sdk.Datum(
        model_input=sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": [int(t) for t in tokens[1:]], "weights": [float(w) for w in weights[1:]]},
    )


# ---------------------------------------------------------------- session

def api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY", "")
    if not key:
        sys.exit("Set FIREWORKS_API_KEY.")
    return key


def connect():
    return _connect(api_key())


def snapshot_path(ref: str | None) -> str | None:
    """`latest` or a checkpoint name (fw-ep2) -> its path from the last train run."""
    if not ref or "/" in ref or ":" in ref:
        return ref
    if not RUN_FILE.exists():
        sys.exit(f"No training run yet ({RUN_FILE.name} is missing): run `python finetune/serverless.py train` first.")
    snaps = json.loads(RUN_FILE.read_text(encoding="utf-8"))["snapshots"]
    if not snaps:
        sys.exit("The last training run saved no checkpoints.")
    name = sorted(snaps)[-1] if ref == "latest" else ref
    if name not in snaps:
        sys.exit(f"No checkpoint {name!r}; have: {', '.join(sorted(snaps))}")
    print(f"snapshot {name}: {snaps[name]}")
    return snaps[name]


def make_sampler_runner(base_model: str, snapshot: str | None, max_tokens: int):
    """eval.py runner: row -> (answer, ms, usage), sampling the base model or a snapshot."""
    from fireworks.training import sdk

    chat = chat_for(base_model)
    service = connect()
    sampler = (service.create_sampling_client(model_path=snapshot) if snapshot
               else service.create_sampling_client(base_model=base_model))
    # Text stop: integer stop ids would need an HF tokenizer object inside the SDK.
    params = sdk.SamplingParams(max_tokens=max_tokens, temperature=0.0, stop=[END])

    def run(row: dict[str, Any]) -> tuple[str, float, dict[str, int]]:
        prompt = chat.prompt_ids(row["messages"][:2])
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                res = sampler.sample(prompt=sdk.ModelInput.from_ints(prompt), num_samples=1,
                                     sampling_params=params)
                res = res.result() if hasattr(res, "result") else res
                ms = (time.perf_counter() - t0) * 1000
                out = [int(t) for t in res.sequences[0].tokens]
                return chat.decode(out), ms, {"prompt_tokens": len(prompt), "completion_tokens": len(out)}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return f"[error] {type(e).__name__}: {e}"[:300], 0.0, {}
                time.sleep(2 ** attempt)
        return "[error]", 0.0, {}

    return run


# ---------------------------------------------------------------- commands

def _rows(split: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATA / f"{split}_eval.jsonl").open(encoding="utf-8")]


def cmd_check(args: argparse.Namespace) -> None:
    """Three answers from the base model: proves access, credits and the template."""
    run = make_sampler_runner(args.base_model, None, 120)
    failed = 0
    for row in _rows("test_real")[:3]:
        text, ms, use = run(row)
        failed += text.startswith("[error")
        print(f"Q: {row['meta']['question']}\nA: {text}\n   ({ms:.0f} ms, {use})\n")
    print("Serverless sampling FAILED: see the errors above." if failed else "Serverless sampling works.")


def _load_train(limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (DATA / "train_sft.jsonl").open(encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def cmd_train(args: argparse.Namespace) -> None:
    from fireworks.training import sdk

    chat = chat_for(args.base_model)
    rows = _load_train(args.limit_train, args.seed)
    seqs = [s for r in rows if (s := sft_tokens(chat, r["messages"])) is not None]
    tokens = sum(len(t) for t, _ in seqs)
    steps_per_epoch = -(-len(seqs) // args.batch)
    price = PRICES.get(args.base_model, (0, 0, 0))[2]
    est = tokens * args.epochs / 1e6 * price
    print(f"{len(seqs)} examples, {tokens:,} tokens/epoch, {args.epochs} epochs, "
          f"{steps_per_epoch * args.epochs} optimizer steps, LoRA rank {args.rank}, lr {args.lr}")
    print(f"Estimated training cost: ${est:.2f} (train meter ${price}/1M tokens; check the pricing page)")
    if not args.yes and input("Start training? [y/N] ").strip().lower() != "y":
        sys.exit("Not started.")

    service = connect()
    trainer = service.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    session = getattr(service, "training_session_name", None) or getattr(service, "training_session_id", None)
    print(f"session: {session}")
    run_log: dict[str, Any] = {"base_model": args.base_model, "session": session, "rank": args.rank,
                               "lr": args.lr, "epochs": args.epochs, "batch": args.batch,
                               "examples": len(seqs), "tokens_per_epoch": tokens, "snapshots": {}, "loss": []}
    RESULTS.mkdir(exist_ok=True)

    def save_log() -> None:
        RUN_FILE.write_text(json.dumps(run_log, indent=1), encoding="utf-8")

    datums = [to_datum(t, w) for t, w in seqs]
    adam = sdk.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                          grad_clip_norm=1.0)
    rng = random.Random(args.seed)
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            batch = [datums[i] for i in order[start:start + args.batch]]
            fb = trainer.forward_backward(batch, "cross_entropy").result()
            trainer.optim_step(adam).result()
            step += 1
            m = getattr(fb, "metrics", None) or {}
            loss = (float(m["loss:sum"]) / max(float(m.get("response_tokens") or 1.0), 1.0)
                    if "loss:sum" in m else None)
            run_log["loss"].append([step, loss])
            if step == 1 or step % 10 == 0:
                print(f"epoch {epoch}/{args.epochs} step {step} loss "
                      f"{'n/a' if loss is None else f'{loss:.4f}'} ({time.time() - t0:.0f} s)")
        # Sampler checkpoint per epoch (names <= 17 chars or promotion fails).
        name = f"fw-ep{epoch}"
        path = trainer.save_weights_for_sampler(name).result().path
        run_log["snapshots"][name] = path
        save_log()
        print(f"saved {name}: {path}")
    print(f"\nDone: {step} steps in {(time.time() - t0) / 60:.1f} min. Snapshots in {RUN_FILE}.")
    print("The session stays open so the checkpoints can be promoted (idle sessions aren't billed).")
    print("Next: evaluate with eval.py --system serverless --snapshot latest, then `serverless.py promote`.")


def cmd_promote(args: argparse.Namespace) -> None:
    """Register a sampler checkpoint as a Fireworks model (needed to deploy it later)."""
    from fireworks.training.sdk import FireworksClient

    run = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    session = run["session"]
    if not str(session).startswith("accounts/"):
        sys.exit(f"Session {session!r} isn't a full resource name; promote it from the Fireworks dashboard.")
    fw = FireworksClient(api_key=api_key())
    ckpts = [c for c in fw.list_training_session_checkpoints(session) if c.get("promotable")]
    want = args.checkpoint or sorted(run["snapshots"])[-1]
    match = [c for c in ckpts if c["name"].rsplit("/", 1)[-1] == want]
    if not match:
        sys.exit(f"No promotable checkpoint {want!r}; have: {[c['name'].rsplit('/', 1)[-1] for c in ckpts]}")
    out = fw.promote_session_checkpoint(name=match[0]["name"], output_model_id=args.model_id,
                                        base_model=run["base_model"])
    print(json.dumps(out, indent=1)[:800])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--limit-train", type=int, default=0, help="first N shuffled rows (0 = all)")
    t.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser("promote")
    p.add_argument("--as", dest="model_id", required=True, help="new model id, e.g. first-witness-call-q36")
    p.add_argument("--checkpoint", help="checkpoint name (default: the last epoch)")
    p.set_defaults(fn=cmd_promote)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
