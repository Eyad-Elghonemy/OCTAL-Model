"""
CustomerData.py

Pydantic schema defining the full set of vehicle features required
for a direct prediction request (/predict/linear), and reused as the
final validation layer for VIN-based predictions.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal


class CustomerData(BaseModel):

    make: Literal[
        'Volkswagen', 'Lexus', 'Subaru', 'Cadillac', 'Toyota', 'Land Rover',
        'Mazda', 'Ram', 'Chrysler', 'GMC', 'Volvo', 'Audi', 'Chevrolet',
        'Tesla', 'Hyundai', 'Ford', 'Porsche', 'Acura', 'Nissan', 'Kia',
        'Jeep', 'BMW', 'Dodge', 'Mercedes-Benz', 'Honda'
    ] = Field(description="Car's Manufacturer/Brand")

    model: Literal[
        'Jetta', 'RX', 'Crosstrek', 'Lyriq', 'Highlander', 'Defender',
        'Mazda3', 'Atlas', '2500', '300', 'Yukon', 'XT5', 'Range Rover',
        'S60', 'Camry', 'Q5', 'Silverado', 'Model 3', '3500', 'Sonata',
        'Camaro', 'Explorer', '911', 'MDX', 'Sentra', 'Mustang',
        'Discovery', 'R8', 'Forte', 'Equinox', 'Model Y', '1500',
        'Grand Cherokee', 'Mazda6', 'M3', 'Malibu', 'Pacifica', 'Panamera',
        'Pathfinder', 'V60', 'A6', 'Sorento', 'Cherokee', 'IS', 'Macan',
        '3 Series', 'Durango', 'Titan', 'Tiguan', 'TLX', 'S-Class',
        'E-Class', 'Tucson', 'RDX', 'Corolla', 'RAV4', 'F-150', 'X5',
        'Outback', 'Cayenne', 'Rogue', 'Accord', 'Acadia', 'ES',
        'Sportage', 'Tacoma', 'CT5', 'Sierra', 'Wrangler', 'Altima',
        'Model S', 'GLE', 'NX', 'Tahoe', 'Q7', 'Integra', 'Impreza',
        'Terrain', 'Civic', 'XC60', 'X3', 'Golf', 'Escape', 'XC90',
        'CX-9', 'Focus', 'CX-5', 'Odyssey', 'Elantra', 'Model X', 'GX',
        'Challenger', 'Telluride', 'Pilot', 'Charger', 'Optima',
        'Escalade', 'CR-V', 'Passat', 'Santa Fe', 'A4', 'C-Class',
        'Forester', 'Compass', '5 Series'
    ] = Field(description="Car's Model")

    year: int = Field(description="Manufacturing Year", ge=1990, le=2025)

    mileage: int = Field(description="Total Mileage In Miles", ge=500, le=300000)

    transmission: Literal['Manual', 'Automatic'] = Field(
        description="Transmission Type"
    )

    fuel_type: Literal['Electric', 'Gasoline', 'Diesel'] = Field(
        description="Fuel Type"
    )

    drivetrain: Literal['RWD', 'FWD', 'AWD'] = Field(
        description="Drivetrain Type"
    )

    body_type: Literal[
        'Sedan', 'SUV', 'Hatchback', 'Pickup Truck', 'Coupe', 'Minivan', 'Wagon'
    ] = Field(description="Body Type")

    trim: Literal['EX', 'LX', 'Touring', 'Base', 'Sport', 'Limited'] = Field(
        description="Trim Level"
    )

    engine_capacity: float = Field(
        description=(
            "Engine Capacity In Liters. Use 0 for Electric vehicles "
            "(no internal combustion displacement)."
        ),
        ge=0.0, le=4.0
    )

    @model_validator(mode="after")
    def _validate_engine_capacity_by_fuel_type(self):
        """
        Electric vehicles have no engine displacement, so engine_capacity
        is allowed to be 0. Every other fuel type must report a real,
        non-zero displacement (>= 0.5L), otherwise the value is almost
        certainly missing/invalid data rather than a genuine reading.
        """
        if self.fuel_type == "Electric":
            if self.engine_capacity != 0:
                raise ValueError(
                    "قيمة سعة المحرك يجب أن تساوي 0 للمركبات الكهربائية."
                )
        else:
            if self.engine_capacity < 0.5:
                raise ValueError(
                    "قيمة سعة المحرك يجب ألا تقل عن 0.5 لتر للمركبات غير الكهربائية."
                )
        return self
