from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "shopping_analysis.pkl"
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
num_imputer = artifact["num_imputer"]
cat_imputer = artifact["cat_imputer"]
encoder = artifact["encoder"]
scaler = artifact["scaler"]
NUM_COLS = list(artifact["num_cols"])
FORM_NUM_COLS = [column for column in NUM_COLS if column != "Response_ID"]
CAT_COLS = list(artifact["cat_cols"])
FEATURE_NAMES = list(artifact["feature_names"])
SELECTED_FEATURES = list(artifact["selected_features"])
SELECTED_FEATURE_INDICES = [FEATURE_NAMES.index(name) for name in SELECTED_FEATURES]

app = Flask(__name__)


def build_form_fields(values=None):
    values = values or {}
    fields = []

    numeric_defaults = dict(zip(NUM_COLS, num_imputer.statistics_))
    for column in FORM_NUM_COLS:
        default = numeric_defaults[column]
        fields.append(
            {
                "name": column,
                "kind": "number",
                "value": values.get(column, str(default)),
            }
        )

    for column, default, categories in zip(
        CAT_COLS, cat_imputer.statistics_, encoder.categories_
    ):
        options = [str(value) for value in categories]
        if column == "Timestamp":
            fields.append(
                {
                    "name": column,
                    "kind": "text",
                    "value": values.get(column, str(default)),
                }
            )
        else:
            fields.append(
                {
                    "name": column,
                    "kind": "select",
                    "value": values.get(column, str(default)),
                    "options": options,
                }
            )

    return fields


def predict_shopping_mode(form):
    numeric_values = []
    for column in NUM_COLS:
        if column == "Response_ID":
            numeric_values.append(float(num_imputer.statistics_[NUM_COLS.index(column)]))
            continue
        value = form.get(column, "").strip()
        if not value:
            raise ValueError(f"{column} is required.")
        try:
            numeric_values.append(float(value))
        except ValueError as exc:
            raise ValueError(f"{column} must be a number.") from exc

    categorical_values = []
    for column in CAT_COLS:
        value = form.get(column, "").strip()
        if not value:
            raise ValueError(f"{column} is required.")
        categorical_values.append(value)

    numeric_array = pd.DataFrame([numeric_values], columns=NUM_COLS)
    categorical_array = pd.DataFrame([categorical_values], columns=CAT_COLS)

    numeric_array = num_imputer.transform(numeric_array)
    numeric_array = scaler.transform(pd.DataFrame(numeric_array, columns=NUM_COLS))
    categorical_array = cat_imputer.transform(categorical_array)
    categorical_array = encoder.transform(
        pd.DataFrame(categorical_array, columns=CAT_COLS)
    )
    if hasattr(categorical_array, "toarray"):
        categorical_array = categorical_array.toarray()

    transformed = np.hstack((numeric_array, categorical_array))
    selected = transformed[:, SELECTED_FEATURE_INDICES]
    return str(model.predict(selected)[0])


@app.get("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    values = request.form.to_dict() if request.method == "POST" else None

    if request.method == "POST":
        try:
            prediction = predict_shopping_mode(request.form)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        fields=build_form_fields(values),
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    print("Shopping Mode Analysis is running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)