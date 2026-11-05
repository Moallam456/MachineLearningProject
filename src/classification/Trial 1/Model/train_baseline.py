"""Decision Tree baseline using shuffled splits of the final filtered dataset.

Run from Trial 1: python run_trial.py
Final test scoring: python run_trial.py --evaluate-test
"""
import argparse
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from Processing.preprocessing import FEATURES, TARGET, build_preprocessor, load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_model():
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])


def evaluate(model, X, y, split_name, out_dir):
    pred = model.predict(X)
    classes = list(model.named_steps["classifier"].classes_)
    auc = float("nan")
    if y.nunique() == 2 and 1 in classes:
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, classes.index(1)]))
    results = {
        "model": "DecisionTree_baseline", "split": split_name, "rows": len(y),
        "n_original_features": len(FEATURES), "canceled_rows": int(y.sum()),
        "cancellation_rate": float(y.mean()), "accuracy": float(accuracy_score(y, pred)),
        "always_zero_accuracy": float((y == 0).mean()),
        "precision_1": float(precision_score(y, pred, zero_division=0)),
        "recall_1": float(recall_score(y, pred, zero_division=0)),
        "f1_1": float(f1_score(y, pred, zero_division=0)), "roc_auc": auc,
    }
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    print(f"\n===== {split_name.title()} =====")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("Confusion matrix (rows=actual, columns=predicted; order=[0, 1]):")
    print(matrix)
    report = classification_report(y, pred, labels=[0, 1], zero_division=0)
    print(report)
    pd.DataFrame([results]).to_csv(out_dir / f"classification_{split_name}_metrics.csv", index=False)
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(
        out_dir / f"classification_{split_name}_confusion_matrix.csv"
    )
    (out_dir / f"classification_{split_name}_report.txt").write_text(report, encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    split_dir = PROJECT_ROOT / "Excel Sheet Inputs"
    parser.add_argument("--train", type=Path, default=split_dir / "H2_training_70.xlsx")
    parser.add_argument("--validation", type=Path, default=split_dir / "H2_validation_15.xlsx")
    parser.add_argument("--test", type=Path, default=split_dir / "H2_testing_15.xlsx")
    parser.add_argument("--evaluate-test", action="store_true", help="Score test data after model selection is finished")
    args = parser.parse_args()
    X_train, y_train = load_split(args.train)
    X_val, y_val = load_split(args.validation)
    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both target classes")
    print(f"Training: {len(y_train):,} rows, {y_train.mean():.2%} cancellations")
    print(f"Validation: {len(y_val):,} rows, {y_val.mean():.2%} cancellations")
    print(f"Predictors: {len(FEATURES)}; target: {TARGET} (excluded from predictors)")
    if abs(y_train.mean() - y_val.mean()) > 0.1:
        print("WARNING: Cancellation rates differ greatly between splits; inspect the split design.")
    print("WARNING: ADR and other mutable inputs need booking-time availability checks.")
    print("NOTE: The source was IQR-filtered before splitting; scores describe that filtered population.")
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, PROJECT_ROOT / "Model" / "decision_tree_baseline.joblib")
    joblib.dump(model.named_steps["preprocessor"], PROJECT_ROOT / "Processing" / "fitted_preprocessor.joblib")
    out_dir = PROJECT_ROOT / "Outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_accuracy = float(accuracy_score(y_train, model.predict(X_train)))
    print(f"Training accuracy: {train_accuracy:.4f}")
    evaluate(model, X_train, y_train, "training", out_dir)
    results = evaluate(model, X_val, y_val, "validation", out_dir)
    pd.DataFrame([{
        **results, "train_rows": len(y_train), "validation_rows": len(y_val),
        "train_accuracy": train_accuracy, "val_accuracy": results["accuracy"],
    }]).to_csv(out_dir / "classification_decision_tree_baseline.csv", index=False)
    if args.evaluate_test:
        X_test, y_test = load_split(args.test)
        evaluate(model, X_test, y_test, "test", out_dir)
    else:
        print("Test evaluation reserved for final assessment; enable with --evaluate-test.")
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
