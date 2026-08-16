from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class ModelInfo(BaseModel):
    name: str = "Mammo-CLIP"
    checkpoint: str = "shawn24/Mammo-CLIP"
    model_type: str = "vision-language foundation model"
    task: str = "image-text retrieval"
    clinical_diagnosis: bool = False
    calibrated_probability: bool = False
    license: str = "CC BY-NC-SA 4.0 (Non-Commercial Research Only)"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    checkpoint: str
    device: str
    error: Optional[str] = None

class ConceptScore(BaseModel):
    concept: str
    score: float
    score_type: str = "zero_shot_text_alignment"

class ViewPrediction(BaseModel):
    view_position: str
    sop_instance_uid: Optional[str] = None
    filename: Optional[str] = None
    predictions: List[ConceptScore]

class StudyInfo(BaseModel):
    study_instance_uid: str
    modality: str = "MG"

class PredictResponse(BaseModel):
    success: bool
    model: ModelInfo
    study: StudyInfo
    views: List[ViewPrediction]
    aggregate_predictions: Optional[List[ConceptScore]] = None
    aggregation_method: Optional[str] = "mean_of_independent_view_scores"
    warnings: List[str]
    processing_time_seconds: float

class ErrorResponse(BaseModel):
    success: bool = False
    error: Dict[str, str]
