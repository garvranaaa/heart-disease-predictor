# ❤️ Heart Disease Predictor

End-to-end ML pipeline — EDA → preprocessing → model comparison → deployed inference. Predicts heart disease risk from 13 clinical features using a trained Random Forest model.

🔗 **Live Demo:** ([https://movie-recommender-garvrana.streamlit.app/](https://heart-disease-predictor-gr.streamlit.app/))

---

## Demo

### Patient Data Input
<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/69563c15-a3d7-4616-95c8-28ced6dace1a" />


### Prediction Output
<img width="1454" height="326" alt="image" src="https://github.com/user-attachments/assets/d896e54c-9c63-457a-837a-46719130dbf5" />

---

## Pipeline

```
heart.csv
  → EDA (class balance, nulls, correlations)
  → 80/20 stratified train/test split
  → StandardScaler (fit on train only — no leakage)
  → Train: Logistic Regression + Random Forest + Gradient Boosting
  → Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  → 5-fold cross-validation
  → Best model deployed via Streamlit
```

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~85% | ~84% | ~87% | ~85% | ~91% |
| **Random Forest** | **~88%** | **~87%** | **~90%** | **~88%** | **~94%** |
| Gradient Boosting | ~87% | ~86% | ~89% | ~87% | ~93% |

Random Forest selected as the best model based on ROC-AUC.

---

## Features

**13 clinical inputs:**

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored (0–3) |
| `thal` | Thalassemia type |

---

## Key ML decisions

**Why stratified split?**
The dataset is ~54% disease / ~46% healthy. Stratifying preserves this ratio in both train and test sets, preventing the model from being evaluated on an unrepresentative subset.

**Why fit the scaler on train only?**
Fitting StandardScaler on the full dataset leaks test set statistics (mean, variance) into training — a form of data leakage that inflates evaluation metrics. The scaler is fit on training data only, then applied identically to the test set.

**Why Random Forest over Logistic Regression?**
LR assumes linear decision boundaries. Heart disease risk involves non-linear interactions between features (e.g. age × chest pain type). RF captures these interactions and consistently outperforms LR on this dataset by ~3% ROC-AUC.

**Why ROC-AUC over Accuracy?**
The dataset is moderately imbalanced. A model that always predicts "disease" would score ~54% accuracy. ROC-AUC measures the model's ability to rank patients correctly regardless of the decision threshold — a more honest metric.

---

## Run Locally

```bash
git clone https://github.com/garvranaaa/heart-disease-predictor
cd heart-disease-predictor

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Mac/Linux

pip install -r requirements.txt

# Train the model first (generates model.pkl, scaler.pkl, results.json, plots)
python train_model.py

# Launch the app
streamlit run app.py
```

Get `heart.csv` from [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and place it in the project folder before running.

---

## Project Structure

```
heart-disease-predictor/
├── app.py                  # Streamlit UI
├── train_model.py          # Training script — run once
├── heart.csv               # Dataset (download from Kaggle)
├── model.pkl               # Saved Random Forest (generated)
├── scaler.pkl              # Saved StandardScaler (generated)
├── results.json            # Per-model metrics (generated)
├── feature_importance.png  # Feature importance chart (generated)
├── confusion_matrix.png    # Confusion matrix (generated)
├── roc_curve.png           # ROC curves (generated)
├── requirements.txt
└── README.md
```

---

## Stack

| Library | Purpose |
|---|---|
| Scikit-learn | Model training, scaling, evaluation |
| Streamlit | Web app and UI |
| Matplotlib | Feature importance, confusion matrix, ROC curves |
| Pandas | Data loading and manipulation |
| NumPy | Array operations |

---

> This tool is for educational purposes only and is not a medical diagnosis.

Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
