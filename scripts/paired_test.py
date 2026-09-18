"""McNemar's test on two quality cells, joined per item.

Why this exists as a script rather than as a line in a notebook. Accuracy on
250 items carries a +/-6pp confidence interval, so two configs can differ by a
real 5 points and still show overlapping intervals -- which invites the reading
"no significant difference" when the correct reading is "the marginal test is
underpowered here". The runs are paired on *identical items*, so the marginal
intervals throw away the pairing that carries the information: only the items
where the two configs disagree say anything at all, and there are usually far
fewer of those than 250.

    python -m scripts.paired_test --suite mmlu_pro_cot --a <cfg> --b <cfg>

Reports, for cells A and B:

  b01  items A got wrong and B got right
  b10  items A got right and B got wrong

Under the null that the two configs are equally accurate, those two counts are
draws from Binomial(b01 + b10, 0.5). Both the chi-square approximation (with
continuity correction) and the exact binomial p are printed, because at these
counts they can differ enough to matter and the exact one is the honest figure.

Read the result against a FLOOR, not against zero. Scoring a config against
*itself* -- identical weights, identical prompts -- does not agree perfectly:
vLLM's batch composition varies run to run, and one different token early in a
long derivation sends the reply down another path. That floor is a measurement,
not a nuisance, and `--floor-note` exists to keep it beside the number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
QUALITY_ROOT = REPO_ROOT / "results" / "quality"


def load(suite: str, config_id: str) -> dict[str, dict]:
    path = QUALITY_ROOT / suite / f"{config_id}.jsonl"
    if not path.exists():
        sys.exit(f"missing cell: {path}")
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["scenario_id"]] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--a", required=True, help="config id of cell A (the baseline)")
    ap.add_argument("--b", required=True, help="config id of cell B (the treatment)")
    ap.add_argument("--floor-note", default="", help="Text printed beside the result")
    args = ap.parse_args()

    a, b = load(args.suite, args.a), load(args.suite, args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("no shared items between the two cells")

    acc_a = sum(a[i]["correct"] for i in shared) / len(shared)
    acc_b = sum(b[i]["correct"] for i in shared) / len(shared)
    agree = sum(a[i]["extracted"] == b[i]["extracted"] for i in shared) / len(shared)

    b01 = sum(1 for i in shared if not a[i]["correct"] and b[i]["correct"])
    b10 = sum(1 for i in shared if a[i]["correct"] and not b[i]["correct"])
    n_disc = b01 + b10

    # Truncation is tracked separately from wrongness throughout this study: a
    # cut-off reply is an UNMEASURED item, not a wrong one, and two configs
    # that truncate at different rates are being compared over different
    # subsets of each reply. Report both rates so the spread is visible here
    # too, not only in the runner's gate.
    tr_a = sum(a[i]["truncated"] for i in shared) / len(shared)
    tr_b = sum(b[i]["truncated"] for i in shared) / len(shared)

    print(f"suite      : {args.suite}   n={len(shared)} shared items")
    print(f"A          : {args.a}   acc={acc_a:.3f}  trunc={tr_a:.3f}")
    print(f"B          : {args.b}   acc={acc_b:.3f}  trunc={tr_b:.3f}")
    print(f"delta      : {acc_b - acc_a:+.3f}   agreement={agree:.3f}")
    print(f"discordant : {n_disc}  (A wrong/B right {b01}, A right/B wrong {b10})")

    if n_disc == 0:
        print("mcnemar    : undefined -- the two cells never disagree on correctness")
        return

    chi2 = (abs(b01 - b10) - 1) ** 2 / n_disc
    p_chi = float(stats.chi2.sf(chi2, 1))
    p_exact = float(stats.binomtest(b01, n_disc, 0.5).pvalue)
    print(f"mcnemar    : chi2={chi2:.3f}  p={p_chi:.4f}   exact binomial p={p_exact:.4f}")
    if args.floor_note:
        print(f"floor      : {args.floor_note}")


if __name__ == "__main__":
    main()
