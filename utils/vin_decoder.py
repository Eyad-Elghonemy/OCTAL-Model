"""
vin_decoder.py

Decodes a Vehicle Identification Number (VIN) via the free NHTSA
vPIC API and translates the raw response into the feature values
used by this project's model, via vin_mapping.json.
"""

import json
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(BASE_DIR, "vin_mapping.json")

with open(MAPPING_PATH, "r") as f:
    VIN_MAPPING = json.load(f)

NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"


def _map_value(category: str, raw_value: str):
    """
    Maps a raw value returned by the NHTSA API to the corresponding
    value used in this project's dataset.

    Returns:
        The mapped value if a mapping exists; otherwise None (covers
        empty values, "Not Applicable", and values missing from the
        mapping table).
    """
    if not raw_value or raw_value == "Not Applicable":
        return None
    return VIN_MAPPING.get(category, {}).get(raw_value)


def _round_to_nearest_half(value: float) -> float:
    """Rounds a value to the nearest multiple of 0.5 (e.g. 2.3 -> 2.5)."""
    return round(value * 2) / 2


def decode_vin(vin: str) -> dict:
    """
    Decodes a VIN using the NHTSA API and translates the result into
    this project's standardized feature values.

    Args:
        vin: A 17-character Vehicle Identification Number.

    Returns:
        A dict with keys: make, model, year, body_type, fuel_type,
        drivetrain, transmission, engine_capacity.
        Fields that couldn't be mapped are returned as None — it's the
        caller's responsibility to decide how to handle missing values.

    Raises:
        ValueError: if the VIN isn't exactly 17 characters, the NHTSA
            request fails, or the VIN can't be decoded at all.
    """
    vin = vin.strip().upper()

    if len(vin) != 17:
        raise ValueError(
            "رقم الشاسيه (VIN) المُدخل غير صحيح؛ يجب أن يتكوّن من 17 حرفًا بالضبط."
        )

    try:
        response = requests.get(NHTSA_URL.format(vin=vin), timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        raise ValueError(
            "تعذّر الاتصال بخدمة فك تشفير رقم الشاسيه (VIN)؛ يُرجى المحاولة مرة أخرى لاحقًا."
        )

    data = response.json()
    results_list = data.get("Results", [{}])
    results = results_list[0] if results_list else {}

    if not results.get("Make"):
        raise ValueError(
            "تعذّر التعرّف على رقم الشاسيه (VIN) المُدخل، يُرجى التأكد من صحته والمحاولة مجددًا."
        )

    extracted = {
        "make": results.get("Make", "").title() if results.get("Make") else None,
        "model": results.get("Model") or None,
        "year": int(results["ModelYear"]) if results.get("ModelYear") else None,
        "body_type": _map_value("body_type", results.get("BodyClass")),
        "fuel_type": _map_value("fuel_type", results.get("FuelTypePrimary")),
        "drivetrain": _map_value("drivetrain", results.get("DriveType")),
        "transmission": _map_value("transmission", results.get("TransmissionStyle")),
        "engine_capacity": (_round_to_nearest_half(float(results["DisplacementL"]))if results.get("DisplacementL") else None),
    }

    return extracted


def validate_mapping_file(CustomerData):
    """
    Development-time consistency check: verifies that every value
    vin_mapping.json maps to actually exists among CustomerData's
    allowed Literal values. Run this whenever the mapping file changes.

    Args:
        CustomerData: The Pydantic model holding the application's
            allowed categorical values.

    Raises:
        ValueError: if any mapped value isn't a valid CustomerData category.
    """
    checks = {
        "body_type": CustomerData.model_fields['body_type'].annotation.__args__,
        "fuel_type": CustomerData.model_fields['fuel_type'].annotation.__args__,
        "drivetrain": CustomerData.model_fields['drivetrain'].annotation.__args__,
        "transmission": CustomerData.model_fields['transmission'].annotation.__args__,
    }

    for category, allowed_values in checks.items():
        mapped_values = set(VIN_MAPPING.get(category, {}).values())
        invalid = mapped_values - set(allowed_values)
        if invalid:
            raise ValueError(
                f"vin_mapping.json contains invalid values for '{category}': {invalid}"
            )

    print("vin_mapping.json is fully consistent with CustomerData.")
