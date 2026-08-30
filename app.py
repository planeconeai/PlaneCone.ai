import os
import time
import logging
import threading
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    HealthResponse,
    ModelInfo,
    PredictResponse,
    StudyInfo,
    ViewPrediction,
    ConceptScore,
    ErrorResponse
)
from preprocessing import parse_dicom_bytes, prepare_image_tensor
from inference import MammoCLIPInferenceEngine, CHECKPOINT_NAME, TARGET_LABELS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("mammo-ai-app")

app = FastAPI(
    title="Planecone Mammography AI Microservice (Mammo-CLIP Research Prototype)",
    description="Research microservice providing zero-shot mammography image-text alignment scores",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import gc

engine = MammoCLIPInferenceEngine.get_instance()

@app.on_event("startup")
def startup_event():
    """Load model in background thread so port binds immediately."""
    def load_in_background():
        logger.info("Starting background model load thread (512MB RAM Safe)...")
        engine.load_model()
    thread = threading.Thread(target=load_in_background, daemon=True)
    thread.start()
    logger.info("Startup complete — model loading in background thread.")

@app.get("/")
@app.head("/")
def root_health_check():
    """Root health check for Render load balancer."""
    return {"status": "ok", "service": "planeconeai-mammo-ai", "mode": engine.mode}

@app.get("/shiva-checking")
def shiva_checking():
    """Diagnostic endpoint checking service status, port, memory mode, and engine state."""
    try:
        return {
            "status": "WORKING",
            "service": "planeconeai-mammo-ai",
            "port": os.getenv("PORT", "8000"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_loaded": getattr(engine, "loaded", False),
            "engine_mode": getattr(engine, "mode", "unknown"),
            "device": getattr(engine, "device", "cpu"),
            "ram_optimization": "ACTIVE_512MB_SAFE",
            "free_tier_compatible": True
        }
    except Exception as err:
        return {
            "status": "WORKING",
            "service": "planeconeai-mammo-ai",
            "error": str(err),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

@app.get("/health", response_model=HealthResponse)
@app.head("/health")
def health_check():
    """Health check endpoint exposing microservice status and model load state."""
    return HealthResponse(
        status="ok" if engine.loaded else "degraded",
        model_loaded=engine.loaded,
        model=f"Mammo-CLIP ({engine.mode})",
        checkpoint=CHECKPOINT_NAME,
        device=engine.device,
        error=engine.error_message
    )

@app.get("/model-info", response_model=ModelInfo)
def model_info():
    """Returns transparent model metadata, task specifications, and licensing terms."""
    return ModelInfo(
        name="Mammo-CLIP",
        checkpoint=CHECKPOINT_NAME,
        model_type="vision-language foundation model",
        task="image-text retrieval / zero-shot text alignment",
        clinical_diagnosis=False,
        calibrated_probability=False,
        license="CC BY-NC-SA 4.0 (Non-Commercial Research Only)"
    )

@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 401: {"model": ErrorResponse}})
async def predict_mammography_study(
    files: List[UploadFile] = File(...),
    x_internal_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """
    Accepts DICOM mammography files (Modality=MG), validates internal authorization headers,
    runs Mammo-CLIP zero-shot text alignment, and returns structured per-view scores.
    """
    expected_key = os.getenv("MAMMO_AI_INTERNAL_KEY", "planecone_secure_internal_key_2026")
    auth_bearer = f"Bearer {expected_key}"

    if os.getenv("MAMMO_AI_REQUIRE_AUTH", "true").lower() == "true":
        if x_internal_api_key != expected_key and authorization != auth_bearer:
            logger.warning("Unauthorized attempt to access /predict endpoint without valid internal key")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"success": False, "error": {"code": "UNAUTHORIZED", "message": "Invalid internal API key or Bearer token."}}
            )

    if not files:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": {"code": "NO_FILES", "message": "No files provided in request."}}
        )

    if not engine.loaded or engine.model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MODEL_UNAVAILABLE", "message": "Mammo AI model is unavailable or failed to load."}}
        )

    t0 = time.time()
    processed_tensors = []
    views_metadata = []
    study_uids = set()

    for file in files:
        try:
            content = await file.read()
            if not content:
                continue

            img_array, meta = parse_dicom_bytes(content, filename=file.filename)
            del content # Immediately release raw bytes from memory
            
            # Validation 1: Check DICOM Modality
            modality = (meta.get("modality") or "").upper()
            if modality and modality != "MG":
                del img_array
                gc.collect()
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error": {
                            "code": "UNSUPPORTED_MODALITY",
                            "message": f"File '{file.filename}' has Modality='{modality}'. Expected mammography DICOM with Modality=MG."
                        }
                    }
                )

            if img_array is None:
                logger.warning(f"Could not extract pixel data from {file.filename}")
                continue

            tensor = prepare_image_tensor(img_array)
            del img_array # Immediately release raw megapixel array
            gc.collect()
            processed_tensors.append(tensor)

            suid = meta.get("study_instance_uid")
            if suid:
                study_uids.add(suid)

            views_metadata.append({
                "view_position": meta.get("view_position", "UNKNOWN"),
                "sop_instance_uid": meta.get("sop_instance_uid"),
                "filename": file.filename
            })
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")

    # Validation 2: Check StudyInstanceUID consistency
    if len(study_uids) > 1:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "MIXED_STUDY_UIDS",
                    "message": f"Uploaded files belong to multiple distinct studies ({list(study_uids)}). All files in a request must share the same StudyInstanceUID."
                }
            }
        )

    if not processed_tensors:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": {"code": "INVALID_IMAGE_DATA", "message": "Failed to extract valid image data from uploaded files."}}
        )

    study_uid = list(study_uids)[0] if study_uids else "UNKNOWN_STUDY"

    try:
        per_view_results, aggregate_scores, elapsed_seconds = engine.predict(processed_tensors, views_metadata)
    except Exception as err:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MODEL_UNAVAILABLE", "message": f"Inference execution failed: {err}"}}
        )

    views_response = [
        ViewPrediction(
            view_position=v["view_position"],
            sop_instance_uid=v["sop_instance_uid"],
            filename=v["filename"],
            predictions=[ConceptScore(**p) for p in v["predictions"]]
        )
        for v in per_view_results
    ]

    aggregate_response = [ConceptScore(**p) for p in aggregate_scores]

    return PredictResponse(
        success=True,
        model=ModelInfo(
            name="Mammo-CLIP",
            checkpoint=CHECKPOINT_NAME,
            model_type="vision-language foundation model",
            task="image-text retrieval / zero-shot text alignment",
            clinical_diagnosis=False,
            calibrated_probability=False,
            license="CC BY-NC-SA 4.0 (Non-Commercial Research Only)"
        ),
        study=StudyInfo(
            study_instance_uid=study_uid,
            modality="MG"
        ),
        views=views_response,
        aggregate_predictions=aggregate_response,
        aggregation_method="mean_of_independent_view_scores",
        warnings=[
            "Research/decision-support output only.",
            "Scores represent relative zero-shot text alignment and are NOT calibrated clinical probabilities, BI-RADS assessments, or pathology diagnoses.",
            "This model output MUST NOT replace radiologist clinical interpretation."
        ],
        processing_time_seconds=elapsed_seconds
    )
