"""
main.py

FastAPI application entry point. Contains route definitions only --
all business logic lives in the `utils` package:
    - utils.inference        -> prediction from a full CustomerData payload
    - utils.vin_prediction    -> prediction resolution from a VIN
    - utils.error_messages    -> Arabic translation of validation errors
    - utils.usage_tracker     -> operation logging (powers GET /logs)

No login/sign-up system exists in this project. Access to the
operation log is gated by a single SECRET_API_KEY sent as the
X-API-Key header -- everything else stays fully public.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import secrets

from utils.inference import predict_new
from utils.config import APP_NAME, VERSION, SECRET_API_KEY, preprocessor, linear_model
from utils.CustomerData import CustomerData
from utils.VinRequestData import VinRequestData
from utils.error_messages import translate_error
from utils.vin_prediction import resolve_vin_prediction
from utils.usage_tracker import log_operation, get_logs


app = FastAPI(title=APP_NAME, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not SECRET_API_KEY or not secrets.compare_digest(api_key, SECRET_API_KEY):
        raise HTTPException(status_code=403, detail="مفتاح الوصول غير صحيح.")
    return api_key


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for request-body validation failures (HTTP 422).

    Replaces FastAPI's default technical error format with a clean,
    Arabic, frontend-friendly response.
    """
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "توجد بيانات غير صحيحة ضمن الطلب المُرسل، يُرجى مراجعة الحقول التالية.",
            "errors": [translate_error(err) for err in exc.errors()],
        },
    )


@app.get("/", tags=["General"])
async def home():
    """Health check / welcome endpoint."""
    return {"Message": f"Welcome To My {APP_NAME} API v{VERSION}"}


@app.get("/logs", tags=["General"])
async def logs(api_key: str = Depends(verify_api_key)) -> list:
    """
    Returns the operation log: every request handled by /predict/linear
    and /predict/from-vin, each as {timestamp, operation_type, success}.
    Requires the X-API-Key header -- no login/sign-up involved.
    """
    return get_logs()


@app.post("/predict/linear", tags=["Models"])
async def predict_linear(data: CustomerData) -> dict:
    """Predicts a vehicle's price from a fully user-provided payload."""
    try:
        result = predict_new(data=data, preprocessor=preprocessor, model=linear_model)
        log_operation("linear", True)
        return result
    except Exception as e:
        log_operation("linear", False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/from-vin", tags=["Models"])
async def predict_from_vin(request: VinRequestData) -> dict:
    """
    Predicts a vehicle's price starting from a VIN.

    The VIN is decoded via NHTSA and cross-checked against the model's
    supported categories. `make`, `model`, and `year` must be reliably
    resolved from the VIN or the prediction is skipped entirely. Other
    fields (`drivetrain`, `engine_capacity`) fall back to a reasonable
    estimate when NHTSA doesn't provide them -- see utils.vin_prediction
    for the full policy, and the `estimated_*` flags in the response.
    """
    try:
        result = resolve_vin_prediction(request, preprocessor, linear_model)
        log_operation("from-vin", result.get("status") == "success")
        return result
    except ValidationError as e:
        log_operation("from-vin", False)
        raise HTTPException(
            status_code=400,
            detail={
                "message": "تعذّر التحقق من صحة البيانات المُستخرجة والمُدخلة.",
                "errors": [translate_error(err) for err in e.errors()],
            },
        )
    except ValueError as e:
        log_operation("from-vin", False)
        # Logical errors: invalid VIN format, NHTSA connection failure, etc.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_operation("from-vin", False)
        raise HTTPException(status_code=500, detail=str(e))