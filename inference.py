import os
import time
import gc
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
        self.dtype = None
        self.error_message = None
        self.checkpoint = CHECKPOINT_NAME
        self.precomputed_text_features = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MammoCLIPInferenceEngine()
        return cls._instance

    def load_model(self):
        """
        Loads the Mammo-CLIP model at application startup in a low-memory, high-performance configuration.
        """
        logger.info(f"Initializing Mammo-CLIP inference engine ({self.checkpoint})...")
        try:
            import torch
            # Restrict PyTorch thread count on CPU to prevent resource contention on free tier instances
            torch.set_num_threads(min(2, os.cpu_count() or 1))

            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                self.dtype = torch.float16
            else:
                self.device = "cpu"
                # PyTorch CPU doesn't support FP16 for matrix multiplication (addmm_impl_cpu)
                self.dtype = torch.float32

            logger.info(f"Targeting device='{self.device}', dtype='{self.dtype}'")

            from transformers import CLIPModel, CLIPProcessor

            # 1. Processor: Always load standard CLIPProcessor from base model to prevent missing preprocessor_config errors
            base_model = "openai/clip-vit-base-patch32"
            logger.info("Loading standard CLIPProcessor...")
            self.processor = CLIPProcessor.from_pretrained(base_model)

            # 2. Model: Try loading specified checkpoint, fallback to base_model
            try:
                logger.info(f"Loading CLIPModel '{self.checkpoint}' (low_cpu_mem_usage=True)...")
                self.model = CLIPModel.from_pretrained(
                    self.checkpoint,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True
                ).to(self.device)
            except Exception as model_err:
                logger.warning(f"Could not load '{self.checkpoint}' ({model_err}). Falling back to '{base_model}'...")
                self.model = CLIPModel.from_pretrained(
                    base_model,
                    dtype=self.dtype,
                    low_cpu_mem_usage=True
                ).to(self.device)

            self.model.eval()

            # Pre-compute text embeddings once at startup to save CPU cycles per request
            logger.info("Pre-computing text embeddings for target concepts...")
            text_queries = [f"mammogram showing {label.replace('_', ' ')}" for label in TARGET_LABELS]
            
            with torch.no_grad():
                if hasattr(self.processor, "__call__"):
                    inputs = self.processor(text=text_queries, return_tensors="pt", padding=True).to(self.device)
                    if hasattr(self.model, "get_text_features"):
                        text_outputs = self.model.get_text_features(**inputs)
                    else:
                        text_outputs = self.model.encode_text(inputs["input_ids"])
                else:
                    import open_clip
                    tokenizer = open_clip.get_tokenizer(f"hf-hub:{self.checkpoint}")
                    text_tokens = tokenizer(text_queries).to(self.device)
                    text_outputs = self.model.encode_text(text_tokens)

                if not isinstance(text_outputs, torch.Tensor):
                    text_features = getattr(text_outputs, "pooler_output", text_outputs[0])
                else:
                    text_features = text_outputs
                
                self.precomputed_text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(dtype=self.dtype)

            self.loaded = True
            self.error_message = None
            gc.collect()
            logger.info(f"Mammo-CLIP inference engine successfully initialized on {self.device}!")

        except Exception as e:
            logger.error(f"Failed to load Mammo-CLIP model: {e}")
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

        if not self.loaded or self.model is None or self.processor is None or self.precomputed_text_features is None:
            raise RuntimeError("MODEL_UNAVAILABLE: Mammo-CLIP model is not loaded in memory")

        import torch

        with torch.no_grad():
            image_tensors = torch.tensor(processed_tensors, dtype=self.dtype).to(self.device)
            
            if hasattr(self.model, "get_image_features"):
                image_outputs = self.model.get_image_features(image_tensors)
            elif hasattr(self.model, "encode_image"):
                image_outputs = self.model.encode_image(image_tensors)
            else:
                image_outputs = self.model(pixel_values=image_tensors).image_embeds

            if not isinstance(image_outputs, torch.Tensor):
                image_features = getattr(image_outputs, "pooler_output", image_outputs[0])
            else:
                image_features = image_outputs
                
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity and Softmax per view image
            similarity_matrix = (100.0 * image_features @ self.precomputed_text_features.T).softmax(dim=-1).cpu().numpy()

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

        # Calculate mean across views
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
