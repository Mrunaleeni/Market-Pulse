"""
Unit tests for the verdict engine.
Run with: python tests/test_verdict.py
"""

def get_verdict(n, r, p):
    """
    Section 3 logic exactly as specced.
    Returns dict: state, headline, detail, color
    color: grey | amber | green | red
    No raw stats (r, p, rho) in headline or detail — only n and countdown.
    """
    if n < 30:
        return {
            "state":    "early",
            "headline": "Too early to tell",
            "detail":   (f"We have {n} days of data — need 30 before this is reliable. "
                         f"Check back in {30 - n} more trading days."),
            "color":    "grey",
        }
    if p >= 0.05:
        return {
            "state":    "no_signal",
            "headline": "No reliable signal",
            "detail":   (f"With {n} days of data, sentiment does not appear "
                          "to predict returns for this view."),
            "color":    "amber",
        }
    if r > 0:
        return {
            "state":    "positive_signal",
            "headline": "Signal detected — positive",
            "detail":   (f"Positive news tends to come with higher returns "
                         f"({n} days analyzed, statistically significant)."),
            "color":    "green",
        }
    return {
        "state":    "negative_signal",
        "headline": "Signal detected — but inverted",
        "detail":   ("Positive news tends to come with LOWER returns here — "
                     "an unusual pattern worth double-checking in the Proof tab."),
        "color":    "red",
    }


# ── Edge-case tests ───────────────────────────────────────────────────────────
cases = [
    # (n,   r,     p,      expected_state,       expected_color)
    (0,    0.0,   1.0,    "early",               "grey"),    # zero data
    (1,    0.5,   0.01,   "early",               "grey"),    # n=1
    (29,   0.8,   0.001,  "early",               "grey"),    # n=29, one short of threshold
    (30,   0.5,   0.051,  "no_signal",           "amber"),   # p just above 0.05
    (30,   0.5,   0.049,  "positive_signal",     "green"),   # p just below 0.05, r>0
    (30,  -0.5,   0.049,  "negative_signal",     "red"),     # p < 0.05, r<0
    (100,  0.3,   0.001,  "positive_signal",     "green"),   # large n, significant, positive
    (100, -0.3,   0.001,  "negative_signal",     "red"),     # large n, significant, negative
    (50,   0.1,   0.5,    "no_signal",           "amber"),   # enough data, not significant
]

FORBIDDEN_IN_LEVEL1 = ["r=", "p=", "rho", "\u03c1", "R\u00b2", "MAE", "p-value", "pearson"]

print("=" * 65)
print("  Verdict engine — edge case tests")
print("=" * 65)

all_pass = True
for n, r, p, exp_state, exp_color in cases:
    v = get_verdict(n, r, p)

    state_ok = v["state"] == exp_state
    color_ok = v["color"] == exp_color

    # Rule 2: no raw stats in headline or detail
    combined = (v["headline"] + " " + v["detail"]).lower()
    no_leak  = not any(kw.lower() in combined for kw in FORBIDDEN_IN_LEVEL1)

    passed = state_ok and color_ok and no_leak
    if not passed:
        all_pass = False

    tag = "PASS" if passed else "FAIL"
    issues = []
    if not state_ok:  issues.append(f"state={v['state']} want={exp_state}")
    if not color_ok:  issues.append(f"color={v['color']} want={exp_color}")
    if not no_leak:   issues.append("raw stat leaked into Level 1 text")

    print(f"  {tag}  n={n:3d}  r={r:+.1f}  p={p:.3f}"
          f"  -> {v['state']:<22} {v['color']:<6}"
          + (f"  ISSUES: {', '.join(issues)}" if issues else ""))

print()
if all_pass:
    print("All 9 tests passed. Verdict engine is correct.")
else:
    print("SOME TESTS FAILED. Fix before wiring to UI.")
print("=" * 65)
