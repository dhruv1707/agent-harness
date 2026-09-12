"""Lifecycle hooks: external commands that watch a child, and may send it back.

The contract is narrow on purpose. Exit 2 is the only code that changes what the harness
does; everything else fails open, because a broken notifier must not cost a research run.
"""

import asyncio
import json
import os
import stat

import pytest

from harness.hooks import (
    BLOCK_EXIT_CODE,
    HookConfig,
    HookResult,
    fire,
    load_hooks,
    objection,
    run_hook,
)


def run(coro):
    return asyncio.run(coro)


def script(tmp_path, name, body):
    """A real executable, because the point of a hook is that it is a real command."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def hook(command, **kwargs):
    return HookConfig(event="subagent_stop", command=str(command), **kwargs)


# ---- configuration ------------------------------------------------------------


def test_no_hooks_file_means_no_hooks_and_no_error(tmp_path):
    assert load_hooks(tmp_path / "absent.toml") == []


def test_hooks_load_in_file_order_with_their_own_timeouts(tmp_path):
    path = tmp_path / "hooks.toml"
    path.write_text(
        """
[[subagent_start]]
command = "echo start"

[[subagent_stop]]
command = "echo first"
timeout = 5

[[subagent_stop]]
command = "echo second"
enabled = false
""",
        encoding="utf-8",
    )
    hooks = load_hooks(path)

    assert [h.event for h in hooks] == ["subagent_start", "subagent_stop", "subagent_stop"]
    assert hooks[1].timeout == 5
    assert hooks[2].enabled is False


def test_an_unknown_event_is_ignored_rather_than_guessed_at(tmp_path):
    path = tmp_path / "hooks.toml"
    path.write_text('[[before_lunch]]\ncommand = "echo hi"\n', encoding="utf-8")

    assert load_hooks(path) == []


# ---- what a hook receives -----------------------------------------------------


def test_the_payload_arrives_as_json_on_stdin(tmp_path):
    out = tmp_path / "seen.json"
    command = script(tmp_path, "capture.sh", f'cat > "{out}"\n')
    payload = {"event": "subagent_stop", "agent_id": "s-p.researcher-1", "is_error": False}

    result = run(run_hook(hook(command), payload))

    assert result.exit_code == 0
    assert json.loads(out.read_text()) == payload


def test_a_disabled_hook_does_not_run(tmp_path):
    marker = tmp_path / "ran"
    command = script(tmp_path, "touch.sh", f'touch "{marker}"\n')

    run(fire("subagent_stop", {}, [hook(command, enabled=False)]))

    assert not marker.exists()


# ---- exit codes ---------------------------------------------------------------


def test_exit_zero_passes_quietly(tmp_path):
    results = run(fire("subagent_stop", {}, [hook(script(tmp_path, "ok.sh", "exit 0\n"))]))

    assert results[0].exit_code == 0
    assert objection(results) is None


def test_exit_two_objects_and_its_stderr_is_the_complaint(tmp_path):
    command = script(
        tmp_path, "block.sh", '>&2 echo "\'NASA-grade\' is not an approved claim"\nexit 2\n'
    )
    results = run(fire("subagent_stop", {}, [hook(command)]))

    assert results[0].blocked is True
    assert objection(results) == "'NASA-grade' is not an approved claim"


def test_any_other_failure_is_logged_and_does_not_block(tmp_path):
    """A hook that crashes is a broken hook, not a considered refusal."""
    results = run(fire("subagent_stop", {}, [hook(script(tmp_path, "bad.sh", "exit 1\n"))]))

    assert results[0].exit_code == 1
    assert results[0].blocked is False
    assert objection(results) is None


def test_a_blocking_hook_with_nothing_to_say_still_says_something(tmp_path):
    results = run(fire("subagent_stop", {}, [hook(script(tmp_path, "mute.sh", "exit 2\n"))]))

    assert objection(results)


# ---- failing open -------------------------------------------------------------


def test_a_hook_that_hangs_is_abandoned(tmp_path):
    command = script(tmp_path, "hang.sh", "sleep 30\n")

    result = run(run_hook(hook(command, timeout=0.2), {}))

    assert result.error and "within" in result.error
    assert result.blocked is False, "a hook that never answered has not objected"


def test_a_missing_command_does_not_end_the_run(tmp_path):
    result = run(run_hook(hook(tmp_path / "nope.sh"), {}))

    assert result.error and not result.blocked


def test_an_unparseable_command_is_reported_not_raised():
    result = run(run_hook(hook('echo "unterminated'), {}))

    assert result.error and "unparseable" in result.error


def test_an_empty_command_is_refused():
    assert run(run_hook(hook("   "), {})).error == "empty command"


def test_hooks_do_not_run_through_a_shell(tmp_path):
    """`shlex.split` and no `shell=True`, so a value the model chose cannot become an
    expansion. The argument arrives as text."""
    out = tmp_path / "arg"
    command = script(tmp_path, "echo-arg.sh", f'printf "%s" "$1" > "{out}"\n')

    run(run_hook(hook(f"{command} '$(whoami)'"), {}))

    assert out.read_text() == "$(whoami)"


# ---- the contract the pool depends on -----------------------------------------


def test_only_the_first_objection_counts():
    first = HookResult("subagent_stop", "a", exit_code=2, stderr="fix the claim", blocked=True)
    second = HookResult("subagent_stop", "b", exit_code=2, stderr="and the bands", blocked=True)

    assert objection([first, second]) == "fix the claim"


def test_block_is_exactly_two():
    assert BLOCK_EXIT_CODE == 2
