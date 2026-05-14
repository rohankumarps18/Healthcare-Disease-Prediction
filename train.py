import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE


df = pd.read_csv("C://Users//User//Downloads//archive//diabetes.csv")


cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].median())


X = df.drop("Outcome", axis=1)
y = df["Outcome"]


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_res,
    y_res,
    test_size=0.2,
    random_state=42
)


lr = LogisticRegression(max_iter=1000)

rf = RandomForestClassifier(
    random_state=42
)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)


lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)


def evaluate(model):
    y_pred = model.predict(X_test)

    return [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred)
    ]


results = {
    "Logistic Regression": evaluate(lr),
    "Random Forest": evaluate(rf),
    "XGBoost": evaluate(xgb)
}


print("\nModel Performance:\n")

for model, scores in results.items():
    print(
        f"{model} -> "
        f"Accuracy: {scores[0]:.3f}, "
        f"F1 Score: {scores[1]:.3f}, "
        f"AUC: {scores[2]:.3f}"
    )


os.makedirs("models", exist_ok=True)


pickle.dump(xgb, open("models/model.pkl", "wb"))

print("\nModel saved successfully in models/model.pkl")