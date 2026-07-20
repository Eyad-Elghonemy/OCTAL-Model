"""
vin_prediction.py

Business logic for the VIN-based prediction flow: decode the VIN,
validate every extracted field against the model's supported
categories, merge it with the user-provided fields, and run the
prediction.

Design decision
----------------
`make`, `model`, and `year` are never guessed -- if any of these are
missing or don't match a category the model was trained on, the
prediction is skipped entirely and the endpoint reports back exactly
which of the three couldn't be resolved via `missing_<field>: true/false`
flags (not a list).

Every other VIN-decoded field (`body_type`, `fuel_type`, `drivetrain`,
`engine_capacity`) is estimated with a simple, explainable fallback
whenever NHTSA doesn't provide a usable value for it -- whether the
field came back empty/unmapped, or wasn't one of this model's known
categories, is handled the same way, so a prediction always goes
through as long as make/model/year are known. Each estimated field is
flagged in the response as `estimated_<field>: true`, so the frontend
can be upfront with the user about which numbers were estimated rather
than decoded. Fields that came reliably from the VIN get `false`.
"""

from .CustomerData import CustomerData
from .VinRequestData import VinRequestData
from .vin_decoder import decode_vin
from .inference import predict_from_vin_data


# Simple, explainable fallback estimates used only when NHTSA doesn't
# report a usable value. These are intentionally basic (most-common /
# most-conservative choices), not model-based predictions.
_DEFAULT_BODY_TYPE_ESTIMATE = "Sedan"
_DEFAULT_FUEL_TYPE_ESTIMATE = "Gasoline"

_DRIVETRAIN_ESTIMATE_BY_BODY_TYPE = {
    "SUV": "AWD",
    "Pickup Truck": "AWD",
    "Wagon": "AWD",
    "Sedan": "FWD",
    "Hatchback": "FWD",
    "Coupe": "FWD",
    "Minivan": "FWD",
}
_DEFAULT_DRIVETRAIN_ESTIMATE = "FWD"

_ENGINE_CAPACITY_ESTIMATE_BY_BODY_TYPE = {
    "SUV": 2.5, "Pickup Truck": 3.5, "Wagon": 2.0,
    "Sedan": 2.0, "Hatchback": 1.5, "Coupe": 2.0, "Minivan": 2.5,
}
_DEFAULT_ENGINE_CAPACITY_ESTIMATE = 2.0


def _is_valid(field_name: str, value) -> bool:
    """
    Checks whether `value` is not None and is one of the allowed
    Literal values defined for `field_name` on CustomerData.
    """
    if value is None:
        return False
    allowed_values = CustomerData.model_fields[field_name].annotation.__args__
    return value in allowed_values


def resolve_vin_prediction(request: VinRequestData, preprocessor, model) -> dict:
    """
    Runs the full VIN -> prediction pipeline.

    Returns:
        On success:
            {"status": "success", "predicted_price": float,
             "estimated_body_type": bool, "estimated_fuel_type": bool,
             "estimated_drivetrain": bool, "estimated_engine_capacity": bool,
             ...all resolved CustomerData fields flattened at the top level}

        When make/model/year can't be reliably resolved:
            {"status": "incomplete", "message": str, ...all fields the
            VIN decoder returned (including None for unresolved ones),
            "missing_make": bool, "missing_model": bool, "missing_year": bool}
    """
    extracted = decode_vin(request.vin)

    allowed_makes = CustomerData.model_fields["make"].annotation.__args__
    allowed_models = CustomerData.model_fields["model"].annotation.__args__

    # Hard requirements -- never estimated. Reported as individual
    # booleans (missing_<field>: true/false) rather than a list, to
    # stay consistent with the estimated_<field> flags used below.
    missing_make = extracted["make"] not in allowed_makes
    missing_model = extracted["model"] not in allowed_models

    year_field = CustomerData.model_fields["year"]
    year_min = next((m.ge for m in year_field.metadata if hasattr(m, "ge")), 2000)
    year_max = next((m.le for m in year_field.metadata if hasattr(m, "le")), 2025)
    missing_year = extracted["year"] is None or not (year_min <= extracted["year"] <= year_max)

    if missing_make or missing_model or missing_year:
        return {
            "status": "incomplete",
            "predicted_price": None,
            "message": (
                "تعذّر تحديد الماركة أو الموديل أو سنة الصنع من رقم الشاسيه بشكل موثوق، "
                "لذلك لم يتم إجراء التنبؤ بالسعر. البيانات التالية تم التعرف عليها بنجاح."
            ),
            **extracted,
            "missing_make": missing_make,
            "missing_model": missing_model,
            "missing_year": missing_year,
        }

    # --- Everything else: estimate instead of failing outright ---
    estimated_body_type = False
    estimated_fuel_type = False
    estimated_drivetrain = False
    estimated_engine_capacity = False

    if not _is_valid("body_type", extracted["body_type"]):
        extracted["body_type"] = _DEFAULT_BODY_TYPE_ESTIMATE
        estimated_body_type = True
    body_type = extracted["body_type"]

    if not _is_valid("fuel_type", extracted["fuel_type"]):
        extracted["fuel_type"] = _DEFAULT_FUEL_TYPE_ESTIMATE
        estimated_fuel_type = True

    if not _is_valid("drivetrain", extracted["drivetrain"]):
        extracted["drivetrain"] = _DRIVETRAIN_ESTIMATE_BY_BODY_TYPE.get(
            body_type, _DEFAULT_DRIVETRAIN_ESTIMATE
        )
        estimated_drivetrain = True

    if extracted["fuel_type"] == "Electric":
        # Electric vehicles genuinely have zero displacement -- this is a
        # correct value, not an estimate.
        extracted["engine_capacity"] = 0.0
    elif extracted["engine_capacity"] is None or not (0.5 <= extracted["engine_capacity"] <= 4.0):
        extracted["engine_capacity"] = _ENGINE_CAPACITY_ESTIMATE_BY_BODY_TYPE.get(
            body_type, _DEFAULT_ENGINE_CAPACITY_ESTIMATE
        )
        estimated_engine_capacity = True

    merged_data = {
        "make": extracted["make"],
        "model": extracted["model"],
        "year": extracted["year"],
        "mileage": request.mileage,
        "transmission": request.transmission,
        "fuel_type": extracted["fuel_type"],
        "drivetrain": extracted["drivetrain"],
        "body_type": extracted["body_type"],
        "trim": request.trim,
        "engine_capacity": extracted["engine_capacity"],
    }

    # Second validation layer: rebuilding the object through Pydantic
    # guarantees nothing invalid reaches the model, even the estimates.
    validated = CustomerData(**merged_data)

    predicted_price = predict_from_vin_data(
        validated.model_dump(), preprocessor, model
    )

    return {
        "status": "success",
        "predicted_price": predicted_price,
        "estimated_body_type": estimated_body_type,
        "estimated_fuel_type": estimated_fuel_type,
        "estimated_drivetrain": estimated_drivetrain,
        "estimated_engine_capacity": estimated_engine_capacity,
        **validated.model_dump(),
    }
