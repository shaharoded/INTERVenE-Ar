"""
evaluate_emr.py
Static, time-aware evaluation for Transform-EMR predictions.

1.  Put the Excel workbook that contains the sheets
        • tokens_df (The full input a model got before generation with all event Admission -> Terminal)
        • predictions (The output dataframe based on inference.py module)
    at the location you set in EXCEL_FILE below.

2.  Adjust TIME_BIAS_H (hours) or OUT_DIR if you need to.

3.  Run   python evaluate_emr.py
    and the script will produce:
        • printed per-event / overall metrics
        •   <OUT_DIR>/evaluation_plots.png
"""

from __future__ import annotations
import os
from typing  import Dict, List

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer




# ────────────────────────── CONFIG ───────────────────────────────────────── #
EXCEL_FILE = r"C:\Users\yonat\CodeProjects\event-prediction-in-diabetes-care\checkpoints\inference_results.xlsx"
OUT_DIR    = r"C:\Users\yonat\CodeProjects\event-prediction-in-diabetes-care\checkpoints\evaluation_plots"
TIME_BIAS_H  = 24.0                                         # "Forgiveness time" -> was the event predicted in the correct time
OUTCOMES = [
    "RELEASE",
    "DEATH",
    "KETOACIDOSIS",
    "KIDNEY_DISORDER",
    "COMA",
    "EYE_DISORDER",
    "NERVOUS_SYSTEM_DISORDER",
    "VASCULAR_DISORDER",
    "OTHER_COMPLICATION",
    "DEMENTIA",
    "CARDIOVASCULAR_DISORDER",
    "ULCER",
    "INFECTION",
    "MUSCULOSKELETAL_COMPLICATION",
    "NEUROVASCULAR_COMPLICATION"
]
# ─────────────────────────────────────────────────────────────────────────── #

def load_dataframes(file: str):
    input_df = pd.read_excel(file, sheet_name="Input Events")
    gen_df   = pd.read_excel(file, sheet_name="Generated Events")

    # Find input events used during generation
    input_events_during_gen = gen_df[gen_df["IsInput"] == 1][["PatientID", "TimePoint"]]

    # Remove these from input_df to get true future events
    input_df = input_df.merge(
        input_events_during_gen.assign(ToDrop=True),
        on=["PatientID", "TimePoint"],
        how="left"
    )
    input_df = input_df[input_df["ToDrop"].isna()].drop(columns=["ToDrop"])
    input_df = input_df.rename(columns={"PositionToken": "Token"})
    # Keep only generated events
    gen_df = gen_df[gen_df["IsInput"] == 0].copy()

    # Keep only patients present in both sets
    common_pids = set(input_df["PatientID"]).intersection(set(gen_df["PatientID"]))
    input_df = input_df[input_df["PatientID"].isin(common_pids)]
    gen_df   = gen_df[gen_df["PatientID"].isin(common_pids)]

    return input_df, gen_df


def evaluate(level: str = "time-aware",
                  time_bias: float = TIME_BIAS_H,
                  out_dir: str = OUT_DIR,
                  verbose: bool = True
                  ) -> Dict[str, Dict]:
    """
    Entry-point to run one of the three evaluations:
    - level = "3class": RELEASE/DEATH/COMPLICATION classification (patient-level)
    - level = "multilabel": exact complication classification (multi-label, patient-level)
    - level = "time-aware": event-level w/ time error (existing logic)
    """
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    # Load dfs
    input_df, gen_df = load_dataframes(EXCEL_FILE)

    if level == "3class":
        outcome_map = {o: "RELEASE" if "RELEASE" in o else
                           "DEATH" if "DEATH" in o else
                           "COMPLICATION"
                       for o in OUTCOMES}
        return evaluate_3class(input_df, gen_df, outcome_map, verbose)

    elif level == "multilabel":
        return evaluate_multilabel(input_df, gen_df, OUTCOMES, verbose)

    elif level == "time-aware":
        tokens_df = input_df.copy()
        tokens_df["RawConcept"] = tokens_df["Token"]
        pred_df = gen_df.copy()
        return evaluate_core(tokens_df, pred_df,
                              time_bias=time_bias,
                              outcomes=OUTCOMES,
                              out_dir=out_dir,
                              verbose=verbose)
    else:
        raise ValueError(f"Invalid evaluation level: {level}")


# ────────────────────────── CORE LOGIC ───────────────────────────────────── #
def evaluate_3class(input_df, gen_df, outcome_map, verbose=True):
    log_prefix = "[evaluate_3class (aligned)]"

    # Map to 3-class outcomes
    input_df["OutcomeGroup"] = input_df["Token"].map(outcome_map).fillna("NONE")
    gen_df["OutcomeGroup"]   = gen_df["Token"].map(outcome_map).fillna("NONE")

    # Collect true/predicted labels per patient
    def get_labels(df):
        return df[df["OutcomeGroup"].isin(["RELEASE", "DEATH", "COMPLICATION"])] \
                 .groupby("PatientID")["OutcomeGroup"].unique()

    true_labels = get_labels(input_df)
    pred_labels = get_labels(gen_df)

    common = true_labels.index.intersection(pred_labels.index)
    y_true = [set(true_labels[pid]) for pid in common]
    y_pred = [set(pred_labels[pid]) for pid in common]

    mlb = MultiLabelBinarizer(classes=["RELEASE", "DEATH", "COMPLICATION"])
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(y_pred)

    # Overall micro-F1
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"{log_prefix}: Micro-F1={f1:.3f}  Acc={acc:.3f}  (n={len(y_true)})")

    # Per-class metrics
    per_class_metrics = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    for i, label in enumerate(mlb.classes_):
        p, r, f1_c, _ = per_class_metrics[0][i], per_class_metrics[1][i], per_class_metrics[2][i], per_class_metrics[3][i]
        if verbose:
            print(f"{log_prefix}: [{label}]  F1={f1_c:.3f}  P={p:.3f}  R={r:.3f}")

    return dict(
        micro_f1=f1,
        micro_acc=acc,
        per_class={
            label: dict(precision=per_class_metrics[0][i],
                        recall=per_class_metrics[1][i],
                        f1=per_class_metrics[2][i])
            for i, label in enumerate(mlb.classes_)
        }
    )


def evaluate_multilabel(input_df, gen_df, complications, verbose=True):
    def get_labels(df):
        return df[df["Token"].isin(complications)].groupby("PatientID")["Token"].unique()
    log_prefix = "[evaluate_multilabel]"
    true_labels = get_labels(input_df)
    pred_labels = get_labels(gen_df)

    common = true_labels.index.intersection(pred_labels.index)
    y_true = [set(true_labels[pid]) for pid in common]
    y_pred = [set(pred_labels[pid]) for pid in common]
    mlb = MultiLabelBinarizer(classes=OUTCOMES)
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        f1 = acc = 0.0
        per_class_metrics = {outcome: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for outcome in OUTCOMES}
    else:
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=range(len(OUTCOMES)))
        per_class_metrics = {OUTCOMES[i]: {"precision": p[i], "recall": r[i], "f1": f1s[i]} for i in range(len(OUTCOMES))}

    if verbose:
        print(f"{log_prefix}: Micro-F1={f1:.3f}  Acc={acc:.3f}  (n={len(y_true)})")
        for outcome, metrics in per_class_metrics.items():
            print(f"{log_prefix} [{outcome}]  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")

    return dict(
        f1=f1,
        accuracy=acc,
        per_class=per_class_metrics
    )


def evaluate_core(tokens_df: pd.DataFrame,
                   pred_df:   pd.DataFrame,
                   *,
                   time_bias: float,
                   outcomes:  List[str] | None,
                   out_dir:   str,
                   verbose:   bool
                   ) -> Dict[str, Dict]:

    id_col, gt_concept_col, pred_concept_col = "PatientID", "RawConcept", "Token"
    time_col_gt, time_col_pred               = "TimePoint", "TimePoint"
    log_prefix = "[evaluate_core]"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    tokens_df = tokens_df[tokens_df[gt_concept_col].isin(outcomes)]
    pred_df   = pred_df[pred_df[pred_concept_col].isin(outcomes)]

    # Prepare for merge
    gt = tokens_df[[id_col, gt_concept_col, time_col_gt]].rename(
        columns={gt_concept_col: "Concept", time_col_gt: "TrueTime"}
    )
    pr = pred_df[[id_col, pred_concept_col, time_col_pred]].rename(
        columns={pred_concept_col: "Concept", time_col_pred: "PredTime"}
    )

    merged = gt.merge(pr, on=[id_col, "Concept"], how="outer", indicator=True)

    both = merged["_merge"] == "both"
    merged.loc[both, "TimeErr"] = merged.loc[both, "PredTime"] - merged.loc[both, "TrueTime"]
    abs_err = merged["TimeErr"].abs()

    merged["TP"] = both & (abs_err <= time_bias)
    merged["FP"] = (merged["_merge"] == "right_only") | (both & ~merged["TP"])
    merged["FN"] = (merged["_merge"] == "left_only")  | (both & ~merged["TP"])

    # Per-event metrics
    metrics: Dict[str, Dict] = {}
    for concept, sub in merged.groupby("Concept"):
        tp, fp, fn = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
        prec = tp / (tp+fp) if tp+fp else 0.
        rec  = tp / (tp+fn) if tp+fn else 0.
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.
        acc  = tp / (tp+fp+fn) if tp+fp+fn else 0.
        metrics[concept] = dict(tp=tp, fp=fp, fn=fn,
                                precision=prec, recall=rec, f1=f1, accuracy=acc,
                                time_errors=sub.loc[sub["TP"], "TimeErr"].tolist())

    # Micro averages
    tp, fp, fn = map(int, (merged["TP"].sum(),
                           merged["FP"].sum(),
                           merged["FN"].sum()))
    overall_micro = dict(
        tp=tp, fp=fp, fn=fn,
        precision=tp/(tp+fp) if tp+fp else 0.,
        recall   =tp/(tp+fn) if tp+fn else 0.,
        f1       =2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        accuracy =tp/(tp+fp+fn) if tp+fp+fn else 0.,
    )

    # MACRO = mean of per-concept metrics
    macro = {}
    for k in ["precision", "recall", "f1", "accuracy"]:
        values = [m[k] for c, m in metrics.items() if not c.startswith("OVERALL")]
        values = [v for v in values if not np.isnan(v)]  # remove NaNs explicitly
        macro[k] = float(np.nanmean(values)) if values else 0.0

        metrics["OVERALL_MICRO"] = overall_micro
        metrics["OVERALL_MACRO"] = macro
# --------------------------------------------------------------------------- #

    if verbose:
        print(f"{log_prefix} ── Evaluation complete ──")
        for c, v in metrics.items():
            if c.startswith("OVERALL"):        # delay the overall lines
                continue
            print(f"{log_prefix} {c:25s}  "
                  f"F1={v['f1']:.3f}  "
                  f"P={v['precision']:.3f}  "
                  f"R={v['recall']:.3f}  "
                  f"Acc={v['accuracy']:.3f}  "
                  f"(n={v['tp']+v['fn']})")
        print("\n{log_prefix} OVERALL  (micro) "
              f"Acc={overall_micro['accuracy']:.3f}  "
              f"F1={overall_micro['f1']:.3f}  "
              f"P={overall_micro['precision']:.3f}  "
              f"R={overall_micro['recall']:.3f}")
        print("{log_prefix} OVERALL  (macro) "
              f"Acc={macro['accuracy']:.3f}  "
              f"F1={macro['f1']:.3f}  "
              f"P={macro['precision']:.3f}  "
              f"R={macro['recall']:.3f}")
    return metrics

# ───────────── tweak the plotting helper so it can also plot accuracy ─────── #
def _plot(metrics: Dict[str, Dict],
          time_bias: float,
          out_dir: str,
          plot_accuracy: bool = True):
    
    if metrics is None:
        print("No metrics to plot (metrics is None).")
        return

    concepts = [c for c in OUTCOMES if c in metrics]
    f1s      = [metrics[c]["f1"]       for c in concepts]
    accs     = [metrics[c]["accuracy"] for c in concepts]

    # Check for all-zero F1s
    if not f1s or all(f == 0.0 for f in f1s):
        print("No metrics to plot (all F1s are zero).")
        return

    order = np.argsort(f1s)[::-1]
    concepts = [concepts[i] for i in order]
    f1s      = [f1s[i] for i in order]
    accs     = [accs[i] for i in order]

    ncols = 3 if plot_accuracy else 2
    fig, axes = plt.subplots(1, ncols,
                             figsize=(8*ncols, 6),
                             constrained_layout=True)

    ax1, ax2 = axes[0], axes[1]

    # ------------ F1 bar ---------------------------------------------------- #
    ax1.bar(range(len(concepts)), f1s)
    ax1.set_xticks(range(len(concepts)))
    ax1.set_xticklabels(concepts, rotation=45, ha="right")
    ax1.set_ylabel("F1-score")
    ax1.set_title("F1 by event")
    ax1.set_ylim(0, 1)
    for x, y in enumerate(f1s):
        ax1.text(x, y+0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    # ------------ signed error boxplot ------------------------------------- #
    time_err = [metrics[c]["time_errors"] or [np.nan] for c in concepts]
    ax2.boxplot(time_err, labels=concepts, showfliers=False)
    ax2.axhline(0,          ls="--", c="r", alpha=.6)
    ax2.axhline(time_bias,  ls="--", c="g", alpha=.4)
    ax2.axhline(-time_bias, ls="--", c="g", alpha=.4)
    ax2.set_ylabel("Signed error (h)")
    ax2.set_title("Time error for TPs")
    ax2.set_xticklabels(concepts, rotation=45, ha="right")

    # ------------ accuracy bar (optional) ---------------------------------- #
    if plot_accuracy:
        ax3 = axes[2]
        ax3.bar(range(len(concepts)), accs, color="tab:green")
        ax3.set_xticks(range(len(concepts)))
        ax3.set_xticklabels(concepts, rotation=45, ha="right")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Accuracy by event")
        ax3.set_ylim(0, 1)
        for x, y in enumerate(accs):
            ax3.text(x, y+0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    out_file = os.path.join(out_dir, "evaluation_plots.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Plots saved → {out_file}")

# ────────────────────────── RUN IF CALLED DIRECTLY ───────────────────────── #
if __name__ == "__main__":
    results_3class = evaluate(level="3class")
    results_multilabel = evaluate(level="multilabel")
    results_time = evaluate(level="time-aware")

    # Only time-aware returns per-concept metrics
    _plot(metrics=results_time, time_bias=TIME_BIAS_H, out_dir=OUT_DIR)
