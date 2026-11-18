"""Evaluation figures + SHAP explainability.  Usage: python -m churn.report"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from . import config as C
from .data import load
from .train import candidate_models

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(C.MODEL_PATH)
    preds = pd.read_csv(C.REPORT_DIR / "test_predictions.csv")
    y, p = preds["y_true"].values, preds["proba"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color="#2563eb")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, color="#2563eb")
    axes[1].axhline(y.mean(), ls="--", color="grey", lw=1, label="base rate")
    axes[1].set(title="Precision–recall (test)", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", color="#2563eb")
    axes[2].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[2].set(title="Calibration (test)", xlabel="Predicted probability", ylabel="Observed churn rate")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "evaluation.png")

    curve = pd.read_csv(C.REPORT_DIR / "profit_curve_train_oof.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["profit"] / 1000, color="#16a34a")
    ax.axvline(bundle["threshold"], ls="--", color="grey")
    ax.set(title="Expected retention-campaign profit vs threshold (train OOF)",
           xlabel="Decision threshold", ylabel="Profit (£k)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "profit_curve.png")

    # SHAP on an uncalibrated tree model of the same family (TreeExplainer is exact & fast).
    X = preds.drop(columns=["y_true", "proba"])
    tree_name = bundle["model_name"] if bundle["model_name"] != "logistic_regression" else "lightgbm"
    X_all, y_all = load()
    X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=C.TEST_SIZE,
                                              stratify=y_all, random_state=C.RANDOM_STATE)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    pipe = candidate_models(pos_weight)[tree_name]
    pipe.fit(X_train, y_train)
    Xt = pipe.named_steps["prep"].transform(X)
    names = pipe.named_steps["prep"].named_steps["encode"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(Xt, columns=names), show=False, max_display=15)
    plt.title("Top churn drivers (SHAP)")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_summary.png", bbox_inches="tight")
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    imp.head(15).round(4).to_csv(C.REPORT_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    print(imp.head(10))


if __name__ == "__main__":
    main()
