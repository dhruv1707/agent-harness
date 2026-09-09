"""Inspect the control plane: `harness prompt`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import MODEL, cache_floor
from .prompt import AssembledPrompt, RunContext, build_effective_system_prompt

RULE = "=" * 78


def _explain(exc: Exception) -> str:
    """Say what actually went wrong, not just the exception class."""
    text = str(exc)
    if "api key" in text.lower() or "api_key" in text.lower():
        return "no GEMINI_API_KEY — put it in .env or export it"
    for attr in ("message", "details"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value:
            return value
    return f"{type(exc).__name__}: {text[:160]}"


class _Counter:
    """Real token counts when the API is reachable, byte counts when it isn't."""

    def __init__(self, enabled: bool = True, model: str = MODEL):
        self.client = None
        self.model = model
        self.note = ""
        if not enabled:
            self.note = "token counting disabled (--no-tokens); showing bytes"
            return
        try:
            from google import genai

            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        except KeyError:
            self.note = "no GEMINI_API_KEY — put it in .env or export it; showing bytes"
        except Exception as exc:
            self.note = f"{_explain(exc)}; showing bytes"

    def tokens(self, text: str) -> int | None:
        if self.client is None or not text:
            return None
        try:
            return self.client.models.count_tokens(
                model=self.model, contents=text
            ).total_tokens
        except Exception as exc:
            self.client = None
            self.note = f"{_explain(exc)}; showing bytes"
            return None

    def measure(self, text: str) -> str:
        count = self.tokens(text)
        if count is None:
            return f"{len(text.encode('utf-8')):,} B"
        return f"{count:,} tok"


def _render(prompt: AssembledPrompt, counter: _Counter) -> None:
    print(RULE)
    print(f" CACHEABLE PREFIX — system_instruction, stable across runs  [{counter.model}]")
    print(RULE)

    marked = False
    for index, layer in enumerate(prompt.layers, start=1):
        if not layer.cacheable and not marked:
            print()
            print("-" * 78)
            print(" ^^^ CACHE BREAKPOINT — below here rides in contents, never cached ^^^")
            print("-" * 78)
            marked = True
        print()
        print(f"=== [{index}] {layer.name}  ({layer.source})  {counter.measure(layer.text)} ===")
        print()
        print(layer.text)

    print()
    print(RULE)
    stable_tokens = counter.tokens(prompt.stable_text)
    print(
        f" TOTAL  cacheable: {counter.measure(prompt.stable_text)}"
        f"   volatile: {counter.measure(prompt.volatile_text)}"
    )

    floor = cache_floor(counter.model)
    if floor is None:
        print(f" CACHE  floor unknown for {counter.model} — cannot verify")
    elif stable_tokens is None:
        print(f" CACHE  floor is {floor:,} tok for {counter.model} — count unavailable")
    elif stable_tokens >= floor:
        print(f" CACHE  OK — prefix {stable_tokens:,} tok clears the {floor:,} tok floor")
    else:
        print(
            f" CACHE  MISS — prefix {stable_tokens:,} tok is BELOW the {floor:,} tok "
            f"floor for {counter.model}."
        )
        print(
            f"        It will be re-billed in full every run. Either grow the prefix by "
            f"{floor - stable_tokens:,} tok or use a model with a lower floor."
        )

    if counter.note:
        print(f" note: {counter.note}")
    print(RULE)


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("prompt", help="Assemble and inspect the system prompt.")
    show.add_argument("--raw", action="store_true", help="Print the request JSON only.")
    show.add_argument("--system-prompt-file", type=Path, help="Override the default stack.")
    show.add_argument("--append-system-prompt", help="Appended last, after the breakpoint.")
    show.add_argument("--custom-system-prompt", help="A job description; extends the stack.")
    show.add_argument("--model", default=MODEL, help=f"Default: {MODEL}")
    show.add_argument("--window", default="last 7 days")
    show.add_argument("--metric", default=None)
    show.add_argument("--run-id", default="inspect")
    show.add_argument("--source", action="append", default=[], help="Repeatable.")
    show.add_argument("--message", default="Write today's scripts.", help="Sample user turn.")
    show.add_argument("--no-tokens", action="store_true", help="Skip the count_tokens calls.")

    args = parser.parse_args()

    override = None
    if args.system_prompt_file:
        override = args.system_prompt_file.read_text(encoding="utf-8")

    prompt = build_effective_system_prompt(
        run_context=RunContext(
            run_id=args.run_id,
            window=args.window,
            metric=args.metric,
            sources=tuple(args.source),
        ),
        override=override,
        custom=args.custom_system_prompt,
        append=args.append_system_prompt,
    )

    if args.raw:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "system_instruction": prompt.system_instruction,
                    "contents": prompt.contents(args.message),
                },
                indent=2,
            )
        )
    else:
        _render(prompt, _Counter(enabled=not args.no_tokens, model=args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
