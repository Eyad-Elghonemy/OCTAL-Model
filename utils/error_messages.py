"""
error_messages.py

Translates Pydantic validation errors into clear, user-facing Arabic
messages for the frontend.

Bidi note
---------
Arabic (RTL) sentences that contain embedded Latin text (field values,
numbers, enum options) can render with a scrambled word order in some
clients, because the Unicode bidirectional algorithm doesn't always
isolate LTR runs correctly inside an RTL sentence on its own.

To fix this, any embedded LTR content (numbers, English literal values)
is wrapped with Unicode directional isolate marks:
    U+2066 (LRI - Left-to-Right Isolate) ... U+2069 (PDI - Pop Directional Isolate)
This explicitly tells the renderer "this chunk is LTR, isolate it",
so the surrounding Arabic punctuation and word order stay correct
regardless of the client used to display the JSON response.
"""

LRI = "\u2066"  # Left-to-Right Isolate
PDI = "\u2069"  # Pop Directional Isolate


def _isolate(value) -> str:
    """Wraps an embedded LTR value (number, English text) so it renders
    correctly inside an RTL Arabic sentence."""
    return f"{LRI}{value}{PDI}"


# Maps internal (English) field names to their Arabic display labels.
FIELD_NAMES_AR = {
    "vin": "رقم الشاسيه",
    "mileage": "الممشى",
    "trim": "درجة التجهيز",
    "transmission": "ناقل الحركة",
    "make": "الماركة",
    "model": "الموديل",
    "year": "سنة الصنع",
    "fuel_type": "نوع الوقود",
    "drivetrain": "نظام الدفع",
    "body_type": "نوع الهيكل",
    "engine_capacity": "سعة المحرك",
}


def translate_error(err: dict) -> str:
    """
    Converts a single Pydantic error dict into a clear Arabic sentence.

    Covers the error types most commonly raised by this project's models:
    missing fields, invalid literal choices, out-of-range numbers,
    wrong data types, and custom model_validator errors.

    Args:
        err: A single error entry as returned by `exc.errors()` from
            either `RequestValidationError` or `pydantic.ValidationError`.

    Returns:
        A ready-to-display Arabic sentence describing the error.
    """
    field = str(err["loc"][-1]) if err.get("loc") else ""
    field_ar = FIELD_NAMES_AR.get(field, field)
    err_type = err.get("type", "")
    ctx = err.get("ctx", {})

    if err_type == "missing":
        return f"حقل «{field_ar}» مطلوب ولم يتم إرساله."

    if err_type == "literal_error":
        allowed = ctx.get("expected", "")
        return (
            f"القيمة المُدخلة في حقل «{field_ar}» غير معتمدة ضمن النظام. "
            f"القيم المسموح بها هي: {_isolate(allowed)}."
        )

    if err_type == "string_too_short":
        return f"حقل «{field_ar}» يجب ألا يقل طوله عن {_isolate(ctx.get('min_length'))} حرفًا."

    if err_type == "string_too_long":
        return f"حقل «{field_ar}» يجب ألا يتجاوز طوله {_isolate(ctx.get('max_length'))} حرفًا."

    if err_type == "greater_than_equal":
        return f"قيمة حقل «{field_ar}» يجب ألا تقل عن {_isolate(ctx.get('ge'))}."

    if err_type == "less_than_equal":
        return f"قيمة حقل «{field_ar}» يجب ألا تتجاوز {_isolate(ctx.get('le'))}."

    if err_type == "greater_than":
        return f"قيمة حقل «{field_ar}» يجب أن تكون أكبر من {_isolate(ctx.get('gt'))}."

    if err_type == "less_than":
        return f"قيمة حقل «{field_ar}» يجب أن تكون أقل من {_isolate(ctx.get('lt'))}."

    if err_type in ("int_type", "int_parsing"):
        return f"حقل «{field_ar}» يجب أن يكون رقمًا صحيحًا."

    if err_type in ("float_type", "float_parsing"):
        return f"حقل «{field_ar}» يجب أن يكون قيمة رقمية."

    if err_type == "string_type":
        return f"حقل «{field_ar}» يجب أن يكون نصًا."

    if err_type == "bool_type":
        return f"حقل «{field_ar}» يجب أن يكون قيمة منطقية (صحيح/خطأ)."

    if err_type == "value_error":
        # Custom @model_validator errors land here. Pydantic prefixes the
        # raw message with "Value error, " — strip that prefix and use
        # our own Arabic message as-is instead of a generic fallback.
        raw_msg = err.get("msg", "")
        if raw_msg.startswith("Value error, "):
            return raw_msg.replace("Value error, ", "", 1)
        return raw_msg or f"القيمة المُدخلة في حقل «{field_ar}» غير صالحة."

    if err_type in ("value_error.missing", "none_required"):
        return f"القيمة المُدخلة في حقل «{field_ar}» غير صالحة."

    if err_type == "json_invalid":
        return "صيغة البيانات المُرسلة غير صحيحة، يجب أن تكون بصيغة JSON سليمة."

    if err_type in ("model_type", "dict_type"):
        return "بيانات الطلب المُرسلة غير مطابقة للصيغة المطلوبة."

    if err_type == "extra_forbidden":
        return f"حقل «{field_ar}» غير معروف ولا يمكن قبوله ضمن الطلب."

    # Fallback for any error type not explicitly handled above.
    return f"القيمة المُدخلة في حقل «{field_ar}» غير صالحة، يُرجى مراجعتها."
