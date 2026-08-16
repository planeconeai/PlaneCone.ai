import os
import time
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger("mammo-inference")

TARGET_LABELS = [
    "mass",
    "suspicious_calcification",
    "architectural_distortion",
    "asymmetry",
    "normal"
]

CHECKPOINT_NAME = os.getenv("MAMMO_CLIP_CHECKPOINT", "shawn24/Mammo-CLIP")

class MammoCLIPInferenceEngine:
    _instance = None

    def __init__(self):
        self.loaded = False
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.error_message = None
        self.checkpoint = CHECKPOINT_NAME

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MammoCLIPInferenceEngine()
        return cls._instance

    def load_model(self):
        """Loads the Mammo-CLIP model at application startup (singleton pattern)."""
        logger.info(f"Initializing Mammo-CLIP inference engine ({self.checkpoint})...")
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            try:
                from transformers import AutoModel, AutoProcessor
                logger.info(f"Loading HuggingFace model '{self.checkpoint}' on {self.device}...")
                self.processor = AutoProcessor.from_pretrained(self.checkpoint)
                self.model = AutoModel.from_pretrained(self.checkpoint).to(self.device)
                self.model.eval()
                self.loaded = True
                logger.info(f"Mammo-CLIP model loaded successfully on {self.device}")
            except Exception as e:
                logger.warning(f"HuggingFace auto-model load failed for '{self.checkpoint}': {e}.")
                self.loaded = False
                self.error_message = str(e)
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch environment: {e}")
            self.loaded = False
            self.error_message = str(e)

    def predict(self, processed_tensors: List[Any], view_metadata: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Runs zero-shot image-text similarity analysis per view.
        Returns:
            per_view_results: List of dicts containing scores per view.
            aggregate_scores: List of dicts with mean alignment scores across views.
            elapsed_seconds: Execution duration.
        """
        start_time = time.time()

        if not self.loaded or self.model is None or self.processor is None:
            raise RuntimeError("MODEL_UNAVAILABLE: Mammo-CLIP model is not loaded in memory")

        import torch
        text_queries = [f"mammogram showing {label.replace('_', ' ')}" for label in TARGET_LABELS]
        inputs = self.processor(text=text_queries, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_tensors = torch.tensor(processed_tensors, dtype=torch.float32).to(self.device)
            image_features = self.model.get_image_features(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity and Softmax per view image
            similarity_matrix = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        per_view_results = []
        for idx, view_meta in enumerate(view_metadata):
            probs = similarity_matrix[idx]
            concept_scores = []
            for label, prob in zip(TARGET_LABELS, probs):
                concept_scores.append({
                    "concept": label,
                    "score": float(round(float(prob), 4)),
                    "score_type": "zero_shot_text_alignment"
                })
            per_view_results.append({
                "view_position": view_meta.get("view_position", "UNKNOWN"),
                "sop_instance_uid": view_meta.get("sop_instance_uid"),
                "filename": view_meta.get("filename"),
                "predictions": concept_scores
            })

        # Calculate mean across views (explicitly labeled as aggregate mean)
        avg_probs = similarity_matrix.mean(axis=0)
        aggregate_scores = []
        for label, prob in zip(TARGET_LABELS, avg_probs):
            aggregate_scores.append({
                "concept": label,
                "score": float(round(float(prob), 4)),
                "score_type": "zero_shot_text_alignment"
            })

        elapsed = round(time.time() - start_time, 3)
        return per_view_results, aggregate_scores, elapsed
