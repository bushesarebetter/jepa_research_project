# ponytail: one-off self-check that the compiled PDF text contains each headline
# number from PAPER_FACTS.md, printed as a (paper claim vs PAPER_FACTS) table.
# Not a permanent test; delete after submission.
import re, sys
from pypdf import PdfReader

reader = PdfReader("main.pdf")
text = "\n".join(p.extract_text() or "" for p in reader.pages)
norm = re.sub(r"\s+", " ", text)

# (headline claim as written in paper, PAPER_FACTS value, substring(s) that must appear)
checks = [
    ("JEPA cell-4 probe, quadrant = 0.51+-0.02",      "0.514+-0.017 (Tab2 -> 0.51)", ["0.51"]),
    ("JEPA cell-4 probe, gridworld = 0.51+-0.02",      "0.508+-0.015 (Tab2 -> 0.51)", ["0.51"]),
    ("JEPA+Reward cell-4 probe = 1.00+-0.00",          "1.00+-0.00",                  ["1.00"]),
    ("jepa_ac gridworld = 0.61+-0.08",                 "0.607+-0.078",                ["0.61"]),
    ("jepa_invdyn gridworld = 0.60+-0.10",             "0.603+-0.105",                ["0.60"]),
    ("MI jepa_reward cell-4 = 0.693 / 0.69 nats",      "0.693+-0.000",                ["0.693", "0.69"]),
    ("MI jepa cell-4 = 0.00 nats",                     "0.000",                       ["0.00"]),
    ("bisim error (Obs 1) = 1.893",                    "1.89300",                     ["1.893"]),
    ("JEPA latent class distance = 0.105",             "0.10523",                     ["0.105"]),
    ("oracle/recon class separation = 1.998",          "1.998 (1.99765/1.99823)",     ["1.998"]),
    ("JEPA retention vs oracle ~ 5%",                  "0.05266",                     ["5\\%", "5%"]),
    ("JEPA cell-4 linear probe (full) = 0.49",         "0.49275",                     ["0.49"]),
    ("min reward fraction passing = 0.02",             "0.02 (mean 0.773)",           ["0.02"]),
    ("min reward 1% accuracy = 0.72",                  "0.7226",                      ["0.72"]),
    ("eff_rank quadrant droppers 38-42",               "38.97-41.98",                 ["38", "42"]),
    ("eff_rank gridworld 2.75-4.61",                   "2.75-4.61",                   ["2.75", "4.61"]),
    ("eff_rank recon gridworld = 3.30",                "3.30+-0.17",                  ["3.30"]),
    ("capacity JEPA 0.510 -> 0.558",                   "0.5096 -> 0.5577",            ["0.510", "0.558"]),
    ("jepa_reward dip @1024 = 0.907 / 0.932",          "0.9069 / 0.9322",             ["0.907", "0.932"]),
    ("switch-color JEPA p=0.5 = 0.61+-0.13",           "0.614+-0.132",                ["0.61"]),
    ("EMA decay = 0.996",                              "0.996",                       ["0.996"]),
]

print("%-44s | %-28s | %s" % ("CLAIM IN PAPER", "PAPER_FACTS VALUE", "MATCH"))
print("-" * 92)
missing = []
for claim, val, subs in checks:
    hit = any(s.replace("\\", "") in norm for s in subs)
    print("%-44s | %-28s | %s" % (claim, val, "OK" if hit else "** MISMATCH **"))
    if not hit:
        missing.append(claim)

print("\nanchor term 'exogenous control-relevant feature' occurrences:",
      norm.lower().count("exogenous control-relevant feature"))
print("verbatim key sentence present:",
      "not irrelevant" in norm and "control-decisive" in norm)
print("PAGES (total):", len(reader.pages))
print("MISMATCHES:", missing if missing else "none")
sys.exit(1 if missing else 0)
