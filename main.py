import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_data(path="train_revised.csv"):
    return pd.read_csv(path)

def basic_preprocess(df):
    df = df.copy()
    target_candidates = [c for c in df.columns if c.lower() in ("seats_sold","seats","target","sales","demand")]
    if len(target_candidates) == 0:
        nums = df.select_dtypes(include=["number"]).columns.tolist()
        if len(nums) == 0:
            raise ValueError("No numeric columns found to use as target")
        target = nums[-1]
    else:
        target = target_candidates[0]
    y = df[target].copy()
    X = df.drop(columns=[target])
    return X, y, target

def build_pipeline(X):
    cats = X.select_dtypes(include=["object","category"]).columns.tolist()
    nums = X.select_dtypes(include=["number","int64","float64"]).columns.tolist()
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, nums),
        ("cat", cat_pipe, cats)
    ], remainder="drop", sparse_threshold=0)
    return pre

def train_and_evaluate(X_train, X_test, y_train, y_test, pre):
    lin = Pipeline([("pre", pre), ("model", LinearRegression())])
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    m_lin = metrics(y_test, y_pred_lin)
    m_rf = metrics(y_test, y_pred_rf)
    print("Linear Regression metrics:", m_lin)
    print("Random Forest metrics:", m_rf)

    best = ("rf", rf, m_rf["rmse"]) if m_rf["rmse"] < m_lin["rmse"] else ("linear", lin, m_lin["rmse"])
    print(f"Best model: {best[0]} (rmse={best[2]:.4f})")
    return lin, rf, best, X_test, y_test

def main():
    df = load_data("train_revised.csv")
    print("Data shape:", df.shape)
    X, y, target = basic_preprocess(df)
    print("Using target column:", target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = build_pipeline(X_train)
    lin, rf, best, X_test, y_test = train_and_evaluate(X_train, X_test, y_train, y_test, pre)
    joblib.dump(best[1], "best_model.pkl")
    print("✅ Saved best model to best_model.pkl")

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    best_model = best[1]
    y_pred = best_model.predict(X_test)

    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted - Transport Demand")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residual Distribution")
    plt.xlabel("Error (y_test - y_pred)")
    plt.show()

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        preprocessor = best_model.named_steps["pre"]
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        feature_names = list(num_features) + list(cat_features)
        importances = best_model.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(8,5))
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title("Top 10 Feature Importances")
        plt.show()

if __name__ == "__main__":
    main()