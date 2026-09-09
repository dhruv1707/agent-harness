"""Inspect the control plane: `harness prompt`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import MODEL
from .prompt import AssembledPrompt, RunContext, build_effective_system_prompt

RULE = "=" * 78


class _Counter:
    """Real token counts when credentials exist, byte counts when they don't."""

    def __init__(self, enabled: bool = True):
        self.client = None
        self.note = ""
        if not enabled:
            self.note = "token counting disabled (--no-tokens); showing bytes"
            return
        try:
            import anthropic

            self.client = anthropic.Anthropic()
        except Exception as exc:  # package trouble, bad config, anything
            self.note = f"token counting unavailable ({type(exc).__name__}); showing bytes"

    def measure(self, text: str) -> str:
        if self.client is None:
            return f"{len(text.encode('utf-8')):,} B"
        try:
            result = self.client.messages.count_tokens(
                model=MODEL,
                system=text,
                messages=[{"role": "user", "content": "."}],
            )
            return f"~{result.input_tokens:,} tok"
        except Exception as exc:
            # The SDK only resolves credentials at request time, so a missing key
            # surfaces here as a TypeError rather than at construction.
            reason = (
                "no API credentials — set ANTHROPIC_API_KEY or run `ant auth login`"
                if "authentication" in str(exc).lower()
                else f"count_tokens failed ({type(exc).__name__})"
            )
            self.client = None
            self.note = f"{reason}; showing bytes"
            return f"{len(text.encode('utf-8')):,} B"


def _render(prompt: AssembledPrompt, counter: _Counter) -> None:
    print(RULE)
    print(" CACHEABLE PREFIX — stable across runs, carries cache_control: ephemeral")
    print(RULE)

    breakpoint_printed = False
    for index, layer in enumerate(prompt.layers, start=1):
        if not layer.cacheable and not breakpoint_printed:
            print()
            print("-" * 78)
            print(" ^^^ CACHE BREAKPOINT — everything below varies per run ^^^")
            print("-" * 78)
            breakpoint_printed = True

        print()
        print(f"=== [{index}] {layer.name}  ({layer.source})  {counter.measure(layer.text)} ===")
        print()
        print(layer.text)

    print()
    print(RULE)
    stable = counter.measure(prompt.stable_text)
    volatile = counter.measure(prompt.volatile_text)
    print(f" TOTAL  cacheable: {stable}   volatile: {volatile}")
    if counter.note:
        print(f" note: {counter.note}")
    print(RULE)


def main() -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("prompt", help="Assemble and inspect the system prompt.")
    show.add_argument("--raw", action="store_true", help="Print the wire JSON blocks only.")
    show.add_argument("--system-prompt-file", type=Path, help="Override the default layer stack.")
    show.add_argument("--append-system-prompt", help="Appended last, after the breakpoint.")
    show.add_argument("--custom-system-prompt", help="A job description; extends the stack.")
    show.add_argument("--window", default="last 7 days")
    show.add_argument("--metric", default=None)
    show.add_argument("--run-id", default="inspect")
    show.add_argument("--source", action="append", default=[], help="Repeatable.")
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
        print(json.dumps(prompt.blocks(), indent=2))
    else:
        _render(prompt, _Counter(enabled=not args.no_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
