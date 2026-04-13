import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def main():
    # Load dataset
    red_path = "dataset/wine+quality/winequality-red.csv"
    white_path = "dataset/wine+quality/winequality-white.csv"

    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    data = pd.concat([red, white], axis=0)

    # Features & target
    X = data.drop("quality", axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=8)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model (IMPORTANT: root folder)
    joblib.dump(model, "model.pkl")

    # Save metrics (IMPORTANT FORMAT)
    metrics = {
        "mse": mse,
        "r2": r2
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Print logs (required for exam)
    print(f"MSE: {mse}")
    print(f"R2: {r2}")


if __name__ == "__main__":
    main()
