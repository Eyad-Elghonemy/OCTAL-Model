"""
inference.py

Runs the trained preprocessing pipeline and regression model to turn
validated vehicle data into a predicted price.
"""

from .CustomerData import CustomerData
import pandas as pd
import numpy as np


def predict_new(data: CustomerData, preprocessor, model) -> dict:
    """
    Predicts a vehicle's price from a fully validated CustomerData object.

    Args:
        data: The validated request payload.
        preprocessor: The fitted preprocessing pipeline used at training time.
        model: The trained regression model.

    Returns:
        {"predicted_price": float} — the predicted price in the
        original (non-log) scale, rounded to two decimal places.
    """
    df = pd.DataFrame([data.model_dump()])

    X_processed = preprocessor.transform(df)

    y_pred_log = model.predict(X_processed)

    y_pred_price = np.expm1(y_pred_log)

    return {
        "predicted_price": round(float(y_pred_price[0]), 2)
    }


def predict_from_vin_data(validated_data: dict, preprocessor, model) -> float:
    """
    Predicts the vehicle price using data that has already been
    validated and merged from two sources:
        - Vehicle information extracted from the VIN.
        - User-provided fields (mileage, trim, transmission).

    Unlike predict_new(), this function expects a plain validated
    dictionary rather than a Pydantic model instance.

    Args:
        validated_data: A validated dictionary containing all required
            features for prediction.
        preprocessor: The fitted preprocessing pipeline used at training time.
        model: The trained regression model.

    Returns:
        The predicted vehicle price, converted back from the
        logarithmic scale and rounded to two decimal places.
    """
    df = pd.DataFrame([validated_data])

    X_processed = preprocessor.transform(df)

    y_pred_log = model.predict(X_processed)

    y_pred_price = np.expm1(y_pred_log)

    return round(float(y_pred_price[0]), 2)
