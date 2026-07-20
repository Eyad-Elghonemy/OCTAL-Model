"""
VinRequestData.py

Pydantic schema for the /predict/from-vin endpoint's request body.

Only fields that genuinely cannot be trusted from the VIN decoder are
collected here: mileage (never encoded in a VIN), trim (no reliable
mapping between NHTSA's series/trim data and this model's trim
categories), and transmission (frequently missing or inconsistent in
NHTSA's response).

Note: the frontend only ever collects these four inputs. There is no
UI path for the user to supply corrections for any other field, so
/predict/from-vin must resolve every other field itself from the VIN,
or fail outright with a clear, final message — it must never ask for
a field the frontend has no way to send.
"""

from pydantic import BaseModel, Field
from typing import Literal


class VinRequestData(BaseModel):

    vin: str = Field(
        description="17-character Vehicle Identification Number",
        min_length=17,
        max_length=17
    )

    mileage: int = Field(description="Total Mileage In Miles", ge=500, le=300000)

    trim: Literal['EX', 'LX', 'Touring', 'Base', 'Sport', 'Limited'] = Field(
        description="Trim Level (not derivable from VIN)"
    )

    transmission: Literal['Manual', 'Automatic'] = Field(
        description="Transmission Type (not reliably derivable from VIN)"
    )
