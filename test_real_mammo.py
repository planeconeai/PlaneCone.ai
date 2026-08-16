#!/usr/bin/env python3
"""
Standalone Real DICOM Model Test Script for Mammo-CLIP Research Prototype.
Loads actual model weights and runs inference against a directory of mammography DICOM files.
Usage:
    python test_real_mammo.py /path/to/dicom_directory
"""

import sys
import os
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("test-real-mammo")

# Ensure planeconeai-mammo-ai module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import parse_dicom_bytes, prepare_image_tensor
from inference import MammoCLIPInferenceEngine, CHECKPOINT_NAME

def run_real_mammo_test(dicom_dir: str):
    logger.info(f"Starting real DICOM test in directory: {dicom_dir}")

    if not os.path.exists(dicom_dir):
        logger.error(f"Directory not found: {dicom_dir}")
        sys.exit(1)

    dicom_files = [
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if f.lower().endswith((".dcm", ".dicom"))
    ]

    if not dicom_files:
        logger.error(f"No DICOM files found in directory: {dicom_dir}")
        sys.exit(1)

    logger.info(f"Found {len(dicom_files)} DICOM files to evaluate.")

    # Step 1: Initialize Singleton Model Engine
    engine = MammoCLIPInferenceEngine.get_instance()
    logger.info("Loading Mammo-CLIP model weights...")
    engine.load_model()

    if not engine.loaded or engine.model is None:
        logger.error(f"MODEL LOAD FAILURE: Could not load Mammo-CLIP model weights. Error: {engine.error_message}")
        print(json.dumps({
            "success": False,
            "error": {
                "code": "MODEL_LOAD_FAILED",
                "message": f"Failed to load Mammo-CLIP model weights: {engine.error_message}"
            }
        }, indent=2))
        sys.exit(1)

    logger.info(f"Model successfully loaded on device: {engine.device}")

    # Step 2: Read and preprocess DICOM files
    processed_tensors = []
    views_metadata = []
    study_uids = set()

    for filePath in dicom_files:
        filename = os.path.basename(filePath)
        logger.info(f"Preprocessing DICOM file: {filename}")
        with open(filePath, "rb") as f:
            content = f.read()

        img_array, meta = parse_dicom_bytes(content, filename=filename)
        if img_array is None:
            logger.warning(f"Skipping {filename}: Could not extract pixel data.")
            continue

        tensor = prepare_image_tensor(img_array)
        processed_tensors.append(tensor)

        suid = meta.get("study_instance_uid")
        if suid:
            study_uids.add(suid)

        views_metadata.append({
            "view_position": meta.get("view_position", "UNKNOWN"),
            "sop_instance_uid": meta.get("sop_instance_uid"),
            "filename": filename
        })

    if not processed_tensors:
        logger.error("No valid image tensors could be extracted from provided DICOM files.")
        sys.exit(1)

    study_uid = list(study_uids)[0] if study_uids else "UNKNOWN_STUDY"

    # Step 3: Run Model Inference
    logger.info("Executing PyTorch zero-shot text alignment inference...")
    t0 = time.time()
    try:
        per_view_results, aggregate_scores, elapsed = engine.predict(processed_tensors, views_metadata)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        print(json.dumps({
            "success": False,
            "error": {
                "code": "INFERENCE_FAILED",
                "message": str(e)
            }
        }, indent=2))
        sys.exit(1)

    output = {
        "success": True,
        "model": {
            "name": "Mammo-CLIP",
            "checkpoint": CHECKPOINT_NAME,
            "device": engine.device,
            "model_type": "vision-language foundation model",
            "task": "zero-shot text alignment",
            "clinical_diagnosis": False,
            "calibrated_probability": False
        },
        "study": {
            "study_instance_uid": study_uid,
            "modality": "MG"
        },
        "views": per_view_results,
        "aggregate_predictions": aggregate_scores,
        "aggregation_method": "independent_view_average",
        "warnings": [
            "Research/decision-support output only.",
            "Scores represent relative zero-shot text alignment and are NOT calibrated clinical probabilities.",
            "Model output must not replace radiologist clinical interpretation."
        ],
        "processing_time_seconds": elapsed
    }

    print("\n================ REAL INFERENCE OUTPUT ================")
    print(json.dumps(output, indent=2))
    print("=======================================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_real_mammo.py <path_to_dicom_directory>")
        sys.exit(1)
    run_real_mammo_test(sys.argv[1])
