"""
basin/verifier/code_executor.py
====================================
Sandboxed execution of LLM-generated code against a HumanEval-style
``check(candidate)`` test function.

Follows the standard HumanEval evaluation pattern (subprocess isolation,
wall-clock timeout, disabled filesystem/process calls) so that untrusted
model completions cannot affect the host process.

Returns an oracle pass/fail outcome plus a coarse *basin key* describing
*how* a failing candidate failed (assertion, exception type, syntax error,
or timeout) — this is the basin label used for the ρ = n_basins − n_eff
computation.
"""

from __future__ import annotations

import faulthandler
import io
import multiprocessing
import os
import platform
import re
import resource
from contextlib import redirect_stderr, redirect_stdout
from typing import Tuple

PASS = "PASS"


def extract_code(text: str) -> str:
    """Pull a Python code block out of a model response.

    Prefers fenced ```python ... ``` / ``` ... ``` blocks; falls back to the
    raw text (some models reply with bare code, no fences).
    """
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return max(fenced, key=len).strip()
    return text.strip()


def _disable_unsafe_builtins() -> None:
    """Best-effort lockdown of filesystem / process / network calls.

    Mirrors the official HumanEval execution harness: the goal is to stop a
    generated program from doing anything destructive while it runs in this
    (already-isolated) subprocess.
    """
    import shutil
    import subprocess as _subprocess

    builtins_module = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    builtins_module["exit"] = None
    builtins_module["quit"] = None

    os.environ["OMP_NUM_THREADS"] = "1"

    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir",
                  "fork", "forkpty", "killpg", "rename", "renames", "truncate",
                  "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
                  "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
        if hasattr(os, name):
            setattr(os, name, None)

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    _subprocess.Popen = None  # type: ignore[assignment]

    try:
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))  # 1 GiB
    except (ValueError, resource.error):
        pass
    if not platform.uname().system == "Darwin":
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (1 << 28, 1 << 28))
        except (ValueError, resource.error):
            pass


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        return "FAIL:SyntaxError"
    if isinstance(exc, AssertionError):
        return "FAIL:AssertionError"
    return f"FAIL:{type(exc).__name__}"


def _run_in_subprocess(program: str, entry_point: str, result_queue) -> None:
    faulthandler.disable()
    _disable_unsafe_builtins()

    exec_globals: dict = {}
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(program, exec_globals)
            if entry_point not in exec_globals:
                result_queue.put((False, "FAIL:NoEntryPoint"))
                return
        result_queue.put((True, PASS))
    except BaseException as e:  # noqa: BLE001 - we want to classify *anything*
        result_queue.put((False, _classify_exception(e)))


def run_check(
    completion_text: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0,
) -> Tuple[bool, str, str]:
    """Execute a candidate completion against a HumanEval ``check()`` block.

    Parameters
    ----------
    completion_text:
        Raw model response (may contain markdown fences / commentary).
    test:
        The dataset's ``test`` field, defining ``check(candidate)``.
    entry_point:
        Name of the function ``check()`` will call.
    timeout:
        Wall-clock seconds before the candidate is marked TIMEOUT.

    Returns
    -------
    (passed, basin_key, code)
        ``passed`` is the oracle pass/fail outcome; ``basin_key`` is one of
        ``"PASS"``, ``"FAIL:<ExceptionType>"``, ``"FAIL:SyntaxError"``,
        ``"FAIL:NoEntryPoint"`` or ``"TIMEOUT"``; ``code`` is the extracted
        source actually executed (for logging/debugging).
    """
    code = extract_code(completion_text)
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_run_in_subprocess, args=(program, entry_point, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return False, "TIMEOUT", code

    if result_queue.empty():
        return False, "FAIL:NoResult", code

    passed, basin_key = result_queue.get()
    return passed, basin_key, code
