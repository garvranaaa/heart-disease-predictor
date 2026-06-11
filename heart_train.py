"""
train_model.py
Run once: python train_model.py
Generates: model.pkl, scaler.pkl, results.json, feature_importance.png,
           confusion_matrix.png, roc_curve.png
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ── 0. Setup ──────────────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE_DIR / "heart.csv")
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target balance:\n{df['condition'].value_counts()}\n")

X = df.drop("condition", axis=1)
y = df["condition"]

# ── 2. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train three models ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

results: dict[str, dict[str, float]] = {}
trained: dict[str, object] = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    trained[name] = model

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    # cast to float before round() to satisfy Pylance
    results[name] = {
        "Accuracy":  round(float(accuracy_score(y_test, y_pred)),  4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall":    round(float(recall_score(y_test, y_pred)),    4),
        "F1 Score":  round(float(f1_score(y_test, y_pred)),        4),
        "ROC-AUC":   round(float(roc_auc_score(y_test, y_prob)),   4),
    }
    print(f"{name}: {results[name]}")

# ── 5. Cross-validation ───────────────────────────────────────────────────────
print("\n5-fold CV ROC-AUC:")
X_sc_full = scaler.fit_transform(X)
for name, model in models.items():
    cv = cross_val_score(model, X_sc_full, y, cv=5, scoring="roc_auc")
    print(f"  {name}: {cv.mean():.4f} (+/- {cv.std():.4f})")

# ── 6. Pick best model ────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["ROC-AUC"])
best_model = trained[best_name]
print(f"\nBest model: {best_name} (ROC-AUC {results[best_name]['ROC-AUC']})")

# ── 7. Save ───────────────────────────────────────────────────────────────────
with open(BASE_DIR / "model.pkl",    "wb") as f: pickle.dump(best_model, f)
with open(BASE_DIR / "scaler.pkl",   "wb") as f: pickle.dump(scaler,     f)
with open(BASE_DIR / "results.json", "w")  as f: json.dump(results,      f)
print("Saved model.pkl, scaler.pkl, results.json")

# ── 8. Feature importance ─────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
if isinstance(best_model, (RFC, GBC)):
    fi = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi.index.tolist(), list(fi.values), color="#4A5A2A")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {best_name}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved feature_importance.png")
    plt.close()

# ── 9. Confusion matrix ───────────────────────────────────────────────────────
from sklearn.base import ClassifierMixin
assert isinstance(best_model, ClassifierMixin)
y_pred_best = best_model.predict(X_test_sc)  # type: ignore[union-attr]
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(5, 4))
colors = [["#D4D9BE", "#B85C3A44"], ["#B85C3A44", "#4A5A2ACC"]]
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color=colors[i][j]))
        ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                ha="center", va="center", fontsize=14, fontweight="600")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Disease", "Disease"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Disease", "Disease"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
ax.set_title(f"Confusion Matrix — {best_name}")
fig.tight_layout()
fig.savefig(BASE_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion_matrix.png")
plt.close()

# ── 10. ROC curve ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
roc_colors = ["#4A5A2A", "#B85C3A", "#6B7A3E"]
for (name, model), c in zip(trained.items(), roc_colors):
    from sklearn.base import ClassifierMixin as CM
    assert isinstance(model, CM)
    y_prob = model.predict_proba(X_test_sc)[:, 1]  # type: ignore[union-attr]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc})", color=c, linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(BASE_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
print("Saved roc_curve.png")
plt.close()

print("\nAll done. Run: streamlit run app.py")