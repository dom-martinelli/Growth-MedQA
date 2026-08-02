"""Recompute the Growth-MedQA leaderboard and subgroups from the released predictions.
Usage: python reproduce.py"""
import pandas as pd, math
pred = pd.read_csv("data/growth_medqa_per_model_predictions.csv")

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(max(0, c-h)*100, 2), round(min(1, c+h)*100, 2))

primary = pred[pred.tier == "primary"]
board = []
for model, g in primary.groupby("model"):
    scored = g[g.predicted_label.notna() & (g.predicted_label.astype(str).str.strip() != "")]
    n = len(scored); k = int(scored.is_correct.sum())
    lo, hi = wilson(k, n)
    board.append((model, n, round(k/n*100, 2), lo, hi))
board = pd.DataFrame(board, columns=["model", "n", "accuracy_pct", "ci95_low", "ci95_high"]).sort_values("accuracy_pct", ascending=False)
print("Primary leaderboard (10 models, 960 items):")
print(board.to_string(index=False))
print("\nSubgroup accuracy with CIs is in supplementary/subgroup_accuracy_wilson_ci.csv")
