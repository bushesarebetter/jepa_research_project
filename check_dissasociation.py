#!/usr/bin/env python
"""Read the quadrant matrix and print the dissociation verdicts in one shot.

Run AFTER:  python quadrant_experiment.py --config matrix --env quadrant --full
Usage:      python check_dissociation.py [path/to/quadrant_matrix.json]
Default:    results/quadrant_matrix.json  (the --full output)

It aggregates per (objective, cell) over seeds and prints acc / MI / effective-rank,
then the four verdicts that decide whether the quadrant story holds.
"""
import json
import os
import sys
from collections import defaultdict

CELL_NAME = {1: "ctrl+rel", 2: "ctrl+irr", 3: "exo+irr", 4: "exo+rel"}
THRESH = 0.75


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "quadrant_matrix.json")
    if not os.path.exists(path):
        smoke = path.replace(".json", "_smoke.json")
        if os.path.exists(smoke):
            print(f"!! {path} not found -- falling back to SMOKE file {smoke}.")
            print("!! These are 150-step/1-seed numbers; run --full before trusting them.\n")
            path = smoke
        else:
            sys.exit(f"no matrix file at {path} (run the --full matrix first)")

    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    objs = data["objectives"]
    cells = data["cells"]
    nseeds = len({r["seed"] for r in results})

    d = defaultdict(lambda: {"acc": [], "mi": [], "rank": []})
    for r in results:
        k = (r["objective"], r["cell"])
        d[k]["acc"].append(r.get("linear_probe_acc"))
        d[k]["mi"].append(r.get("mi_infonce"))
        if r.get("effective_rank") is not None:
            d[k]["rank"].append(r["effective_rank"])

    def acc(o, c):
        return mean(d[(o, c)]["acc"])

    def kept(o, c):
        return acc(o, c) >= THRESH

    print(f"== {path}  (seeds={nseeds}, retain threshold acc>={THRESH}, chance=0.50) ==\n")
    head = f"{'objective':<13}" + "".join(f"{CELL_NAME[c]:>14}" for c in cells)
    print(head)
    print("-" * len(head))
    for o in objs:
        row = f"{o:<13}"
        for c in cells:
            a, mi = acc(o, c), mean(d[(o, c)]["mi"])
            row += f"{a:>5.2f}/{mi:>4.2f}".rjust(14)
        print(row)
    print("\n(cells = acc/MI; MI in nats, max ~0.69; controllable=1,2  exogenous=3,4  "
          "relevant=1,4  irrelevant=2,3)\n")

    # collapse vs selective-drop guard
    low = [(o, c) for o in objs for c in cells
           if o != "oracle" and mean(d[(o, c)]["rank"]) < 5]
    if low:
        print("!! effective_rank < 5 (possible COLLAPSE, not selective drop) at: "
              + ", ".join(f"{o}/{CELL_NAME[c]}" for o, c in low) + "\n")
    elif any(d[(o, c)]["rank"] for o in objs for c in cells):
        print("[ok] effective_rank >= 5 wherever logged (drops are selective, not collapse)\n")
    else:
        print("[note] no effective_rank in this file -- apply Patch 1 and re-run --full.\n")

    print("VERDICTS")
    print("  cell 4 (exo+rel) -- the hard case:")
    for o in objs:
        if (o, 4) in d:
            print(f"     {o:<12} acc={acc(o, 4):.2f}  {'KEEPS' if kept(o, 4) else 'drops'}")
    fix_story = (not kept("jepa", 4)) and (not kept("jepa_ctrl", 4)) and kept("jepa_reward", 4)
    print(f"  => fix story (self-prediction & controllability DROP cell 4, reward KEEPS it): "
          f"{'HOLDS' if fix_story else 'DOES NOT HOLD'}\n")

    if "jepa_reward" in objs:
        sel = (kept("jepa_reward", 1) and kept("jepa_reward", 4)
               and not kept("jepa_reward", 2) and not kept("jepa_reward", 3))
        print(f"  reward fix selective (keeps relevant 1&4, drops irrelevant 2&3): "
              f"{'YES' if sel else 'NO -- it leaks irrelevant features'}")
    if "jepa_ctrl" in objs:
        ctrl = (kept("jepa_ctrl", 1) and kept("jepa_ctrl", 2)
                and not kept("jepa_ctrl", 3) and not kept("jepa_ctrl", 4))
        print(f"  controllability (jepa_ctrl keeps controllable 1&2, drops exogenous 3&4): "
              f"{'YES' if ctrl else 'partial / NO'}")
    if "jepa_ac" in objs:
        diss = kept("jepa_ac", 1) and not kept("jepa_ac", 4)
        print(f"  action-conditioned JEPA dissociation (keeps cell1 ctrl+rel, drops cell4 exo+rel): "
              f"{'YES' if diss else 'NO'}")
    else:
        print("  jepa_ac: NOT in matrix -- add the action-conditioned baseline (Patch 2) to test "
              "the within-JEPA cell1-vs-cell4 dissociation. Plain `jepa` cannot show it.")


if __name__ == "__main__":
    main()