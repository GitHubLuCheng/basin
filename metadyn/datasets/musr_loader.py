"""
MuSR dataset loader.

MuSR (Multi-Step Reasoning) is a harder reasoning benchmark with three tasks:
  - murder_mystery  : identify the culprit from a long narrative
  - object_placements : track object positions through a story
  - team_allocation : allocate people to teams given constraints

Each example is multiple-choice (typically 2-5 options).
Source: TAUR-Lab/MuSR on HuggingFace (https://huggingface.co/datasets/TAUR-Lab/MuSR).

Requirements
------------
  pip install datasets   # for automatic HuggingFace download

Falls back to a local JSONL cache (musr_sample.jsonl in this directory)
if the ``datasets`` library is not installed or the download fails.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterator, List, Optional

from metadyn.datasets.loader import Problem

_PACKAGE_DIR = Path(__file__).parent
_MUSR_CACHE = _PACKAGE_DIR / "musr_sample.jsonl"

MUSR_TASK_NAMES = ["murder_mysteries", "object_placements", "team_allocation"]
_CHOICE_LETTERS = "ABCDE"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_musr_question(narrative: str, question: str, choices: List[str]) -> str:
    """
    Format a MuSR example as a self-contained multiple-choice question string.

    The returned string is used as the ``question`` field of a
    :class:`~metadyn.datasets.loader.Problem` and is inserted verbatim into
    the candidate-generation prompt.
    """
    choice_lines = "\n".join(
        f"  {_CHOICE_LETTERS[i]}. {c.strip()}" for i, c in enumerate(choices)
    )
    return (
        f"{narrative.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Options:\n{choice_lines}\n\n"
        f"Reason step by step through the clues. "
        f"Then write your final answer on the LAST line in EXACTLY this format:\n"
        f"Answer: X\n"
        f"where X is ONE letter from the options above (A, B, C, ...). "
        f"Do not write anything after the Answer line."
    )


def answer_letter(answer_index: int) -> str:
    """Return the choice letter (A, B, C, …) for a zero-based answer index."""
    return _CHOICE_LETTERS[answer_index]


def normalise_musr_answer(raw: str) -> str:
    """
    Extract the first choice letter from a potentially noisy model output.

    Handles:
      "B"  / "b." / "B (Catherine)"  →  "b"
      "Answer: B" / "answer B"       →  "b"
      "**B**" / "- **b**"            →  "b"
      "Option B" / "choice B"        →  "b"
      Verbose last-line fallback: scan *last non-empty line* for a letter
    """
    # Work on the full text for multi-line traces, but prioritise the
    # last non-empty line where the model should put its final answer.
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # Prefer the last line, then the whole text
    candidates = ([lines[-1]] if lines else []) + [raw.strip()]

    for candidate in candidates:
        text = candidate.upper()
        # 1. Explicit "Answer: X" or "Answer X"
        m = re.search(r"ANSWER\s*[:\-–]?\s*\**([A-E])\b", text)
        if m:
            return m.group(1).lower()
        # 2. "Option/Choice/Letter X"
        m = re.search(r"(?:OPTION|CHOICE|LETTER)\s+([A-E])\b", text)
        if m:
            return m.group(1).lower()
        # 3. Markdown bold/italic  **X** or *X*
        m = re.search(r"\*{1,2}([A-E])\*{1,2}", text)
        if m:
            return m.group(1).lower()
        # 4. "- X" / "(X)" / "[X]" bullet at start of line
        m = re.search(r"^[-–•]\s*\(?([A-E])\)?", text.strip())
        if m:
            return m.group(1).lower()
        # 5. Standalone letter anywhere
        m = re.search(r"\b([A-E])\b", text)
        if m:
            return m.group(1).lower()

    return raw.strip().lower()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class MuSRLoader:
    """
    Load MuSR examples as :class:`~metadyn.datasets.loader.Problem` objects.

    Parameters
    ----------
    tasks:
        Which task(s) to include.  Defaults to all three.
    max_problems:
        Truncate to at most this many examples (after shuffling).
    seed:
        Random seed for shuffling.
    cache_path:
        Path to a local JSONL file.  Defaults to the bundled
        ``musr_sample.jsonl`` (if it exists).
    """

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        max_problems: Optional[int] = None,
        seed: int = 42,
        cache_path: Optional[str] = None,
    ) -> None:
        self.tasks = tasks or MUSR_TASK_NAMES
        self.max_problems = max_problems
        self.seed = seed
        self.cache_path = Path(cache_path) if cache_path else _MUSR_CACHE

    # ------------------------------------------------------------------

    def load(self) -> List[Problem]:
        """Return all problems as a list."""
        return list(self.iter_problems())

    def iter_problems(self) -> Iterator[Problem]:
        """Yield shuffled problems up to *max_problems*."""
        problems = list(self._load_raw())
        rng = random.Random(self.seed)
        rng.shuffle(problems)
        for i, p in enumerate(problems):
            if self.max_problems and i >= self.max_problems:
                break
            yield p

    # ------------------------------------------------------------------
    # Internal loaders

    def _load_raw(self) -> List[Problem]:
        """Try HuggingFace, then fall back to local JSONL."""
        try:
            return self._load_from_hf()
        except ImportError:
            pass   # `datasets` not installed
        except Exception:
            pass   # network / format error

        if self.cache_path.exists():
            return list(self._load_from_jsonl(self.cache_path))

        raise FileNotFoundError(
            "MuSR dataset not found.\n"
            "Option 1:  pip install datasets  (downloads automatically)\n"
            f"Option 2:  place a JSONL file at {self.cache_path}\n"
            "           each line: {\"narrative\":\"...\", \"question\":\"...\","
            " \"choices\":[...], \"answer_index\":0, \"task\":\"murder_mystery\"}"
        )

    def _load_from_hf(self) -> List[Problem]:
        from datasets import load_dataset  # type: ignore  # noqa: PLC0415

        problems: List[Problem] = []
        # Dataset has one split per task; no sub-configs.
        # Split names: murder_mysteries, object_placements, team_allocation
        for task in self.tasks:
            ds = load_dataset("TAUR-Lab/MuSR", split=task)
            for i, ex in enumerate(ds):
                narrative  = ex.get("narrative", ex.get("context", ""))
                question   = ex.get("question", "")
                choices    = ex.get("choices", ex.get("answer_choices", []))
                # HF may store choices as a Python-literal string; parse it.
                if isinstance(choices, str):
                    import ast as _ast
                    choices = _ast.literal_eval(choices)
                answer_idx = int(ex.get("answer_index", 0))

                problems.append(Problem(
                    question=format_musr_question(narrative, question, choices),
                    answer=answer_letter(answer_idx),
                    problem_id=f"{task}_{i}",
                ))
        return problems

    def _load_from_jsonl(self, path: Path) -> Iterator[Problem]:
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                narrative  = ex.get("narrative", "")
                question   = ex.get("question", "")
                choices    = ex.get("choices", [])
                answer_idx = int(ex.get("answer_index", 0))
                task       = ex.get("task", "unknown")
                yield Problem(
                    question=format_musr_question(narrative, question, choices),
                    answer=answer_letter(answer_idx),
                    problem_id=ex.get("id", f"{task}_{i}"),
                )
