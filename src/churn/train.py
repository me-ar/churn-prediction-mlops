"""Train, compare and select a churn model; pick a profit-maximising threshold.

Usage:  python -m churn.train
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as C
from .data import download, load
from .features import build_preprocessor


def candidate_models(pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                           class_weight="balanced_subsample", n_jobs=-1,
                                           random_state=C.RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric="logloss",
                                  random_state=C.RANDOM_STATE, n_jobs=-1)),
        ]),
        "lightgbm": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("clf", LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.03,
                                   min_child_samples=30, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, class_weight="balanced",
                                   random_state=C.RANDOM_STATE, verbose=-1)),
        ]),
    }


def expected_profit(y_true, proba, monthly_charges, threshold) -> float:
    """Profit of targeting everyone above `threshold` with a retention offer."""
    target = proba >= threshold
    saved_value = C.RETENTION_SUCCESS_RATE * monthly_charges * C.CUSTOMER_VALUE_MONTHS
    gain = np.where(y_true == 1, saved_value, 0.0) - C.RETENTION_OFFER_COST
    return float(np.sum(gain[target]))


def profit_curve(y_true, proba, monthly_charges):
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [expected_profit(y_true, proba, monthly_charges, t) for t in thresholds]
    return thresholds, np.array(profits)


def classification_metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }


def main() -> None:
    download()
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)

    # 1) Model comparison with 5-fold CV on the training set only.
    comparison = {}
    for name, model in candidate_models(pos_weight).items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        comparison[name] = {"cv_roc_auc": round(roc_auc_score(y_train, oof), 4),
                            "cv_pr_auc": round(average_precision_score(y_train, oof), 4)}
        print(f"{name:22s} ROC-AUC={comparison[name]['cv_roc_auc']:.4f} "
              f"PR-AUC={comparison[name]['cv_pr_auc']:.4f}")
    best_name = max(comparison, key=lambda k: comparison[k]["cv_pr_auc"])
    print(f"Selected: {best_name}")

    # 2) Calibrate the winner (class weighting distorts raw probabilities).
    base = candidate_models(pos_weight)[best_name]
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)

    # 3) Choose a business threshold on out-of-fold calibrated probabilities (no test leakage).
    oof_cal = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thr, profits = profit_curve(y_train.values, oof_cal, X_train["MonthlyCharges"].values)
    threshold = float(thr[int(np.argmax(profits))])
    print(f"Profit-maximising threshold: {threshold:.2f}")

    # 4) Fit on all training data, evaluate once on the held-out test set.
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, proba_test, threshold)
    mc = X_test["MonthlyCharges"].values
    test_metrics["profit_model"] = round(expected_profit(y_test.values, proba_test, mc, threshold), 0)
    test_metrics["profit_target_everyone"] = round(
        expected_profit(y_test.values, np.ones_like(proba_test), mc, 0.5), 0)
    test_metrics["customers_targeted_pct"] = round(float((proba_test >= threshold).mean()), 4)
    print(json.dumps(test_metrics, indent=2))

    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "threshold": threshold,
        "model_name": best_name,
        "features": list(X.columns),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    joblib.dump(bundle, C.MODEL_PATH)
    report = {"selected_model": best_name, "threshold": threshold, "cv_comparison": comparison,
              "test": test_metrics, "n_train": len(X_train), "n_test": len(X_test),
              "assumptions": {"offer_cost": C.RETENTION_OFFER_COST,
                              "success_rate": C.RETENTION_SUCCESS_RATE,
                              "value_months": C.CUSTOMER_VALUE_MONTHS}}
    C.METRICS_PATH.write_text(json.dumps(report, indent=2))

    # Save test predictions for the evaluation/explainability scripts.
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["proba"] = proba_test
    out.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)
    pd.DataFrame({"threshold": thr, "profit": profits}).to_csv(
        C.REPORT_DIR / "profit_curve_train_oof.csv", index=False)
    print(f"Saved model -> {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
