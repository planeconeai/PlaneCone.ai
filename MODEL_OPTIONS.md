# Comprehensive Mammography AI Model Evaluation & Research Matrix

This document provides a verified research evaluation of open mammography AI models, model weights, licensing constraints, commercial usability, and deployment feasibility for Planecone AI.

---

## 1. Candidate Comparison & Licensing Matrix

| Model | Task | Input | Target Output | 4 Views | Pretrained Weights | Code License | Weight / Dataset License | Commercial Use Rights | Clinical Use Rights | API / Docker Server | GPU / Hardware | Status / Verification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mammo-CLIP (BATMAN Lab)** | Zero-Shot Text Alignment | Single 2D DICOM / PNG | Text Alignment Scores | ❌ Independent Single View | ✅ YES (`shawn24/Mammo-CLIP`) | MIT License | **CC BY-NC-SA 4.0** | ❌ **PROHIBITED** (Non-Commercial) | Research Prototype Only | ✅ YES (FastAPI Microservice) | CPU Feasible / GPU (4-8GB VRAM) | **VERIFIED (CURRENT RESEARCH PROTOTYPE)** |
| **MIRAI (MIT CSAIL)** | 5-Year Breast Cancer Risk Prediction | 4-View DICOM (RCC, RMLO, LCC, LMLO) | 1–5 Year Future Cancer Risk Vector | ✅ YES | ✅ YES (`miraimodel/oncoserve`) | MIT License | MGH / EMBED / Karolinska Dataset Terms | **NEEDS COMMERCIAL LICENSE VERIFICATION** | Research / Decision-Support Only (Not FDA Cleared) | ✅ YES (OncoServe Docker) | GPU Recommended (8GB VRAM) | **VERIFIED TASK (FUTURE RISK PREDICTION ONLY)** |
| **VinDr-Mammo Classifiers (VinBigData)** | BI-RADS & Finding Classification | Single / Multi-View DICOM | Mass, Calcification, BI-RADS 1-5 | ✅ YES | ✅ YES (GitHub / PhysioNet) | Apache 2.0 | **PhysioNet Credentialed 1.5.0** | ❌ **PROHIBITED** (PhysioNet Non-Commercial Data Rules) | Research Only | ✅ YES (FastAPI Wrapper) | CPU Feasible / GPU (4GB VRAM) | **VERIFIED (REQUIRES COMMERCIAL DATA LICENSE)** |
| **RSNA Screening Winner Models (Kaggle)** | Binary Malignancy Screening | Multi-View DICOM (CC + MLO) | Malignancy Probability (0.0 to 1.0) | ✅ YES | ✅ YES (Kaggle Checkpoints) | MIT License | **Kaggle RSNA Competition Rules** | ❌ **PROHIBITED** (Non-Commercial Competition Data Rules) | Research Only | ✅ YES (FastAPI Wrapper) | GPU (4-8GB VRAM) | **VERIFIED (NON-COMMERCIAL DATASET RULES)** |
| **Mammo-FM (BATMAN Lab)** | Foundation Representation | Multi-View DICOM | Feature Embeddings | ✅ YES | ✅ YES (Hugging Face) | MIT License | **CC BY-NC-SA 4.0** | ❌ **PROHIBITED** (Non-Commercial) | Research Prototype Only | ❌ NO (Requires API Wrapper) | GPU (8GB VRAM) | **VERIFIED (NON-COMMERCIAL FOUNDATION MODEL)** |
| **Public Cloud APIs (HuggingFace / Replicate)** | Various | Image / DICOM | JSON | Varies | N/A | Varies | Third-Party Cloud Terms | ❌ **NOT SUITABLE FOR CLINICAL / PHI DATA** | ❌ NO | Hosted Cloud | Shared Cloud | **VERIFIED (HIPAA / PHI VIOLATION RISK)** |

---

## 2. Category Breakdown

* **CATEGORY A (Ready-to-use pretrained diagnostic / risk model):**
  1. **MIRAI (MIT CSAIL)** — Pretrained 4-view breast cancer risk prediction container (`miraimodel/oncoserve`). Note: Predicts long-term future cancer risk, NOT immediate lesion location.
  2. **RSNA Competition Winner Models** — Pretrained binary cancer probability models (Subject to Kaggle Non-Commercial rules).
  3. **VinDr-Mammo Community Classifiers** — Pretrained BI-RADS & finding models (Subject to PhysioNet Non-Commercial rules).

* **CATEGORY B (Foundation model requiring downstream fine-tuning):**
  1. **Mammo-CLIP** (`shawn24/Mammo-CLIP`) — Requires linear classification head attached and fine-tuned for calibrated diagnosis.
  2. **Mammo-FM** (`batmanlab/Mammo-FM`) — Requires downstream linear probe or detection head.

* **CATEGORY C (Research code requiring training from scratch):**
  1. **VinDr-Mammo Baseline Training Scripts** — Benchmark training scripts requiring local GPU cluster and annotated data.

---

## 3. Crucial Licensing & Deployment Summary

1. **Current Research Model (`Mammo-CLIP`):** Retained purely as an internal research prototype for zero-shot text alignment visualization. **Must not be deployed commercially** due to CC BY-NC-SA 4.0 terms.
2. **MIRAI (MIT CSAIL):** Represents a promising candidate for multi-view future risk prediction, but **does not output immediate mass/calcification bounding boxes**. Commercial use requires independent verification of dataset licensing terms and regulatory clearance.
3. **Commercially Licensed Finding Models:** If Planecone AI requires calibrated mass/calcification detection and BI-RADS scoring for commercial clinical software, it must either:
   - Secure a commercial data agreement with VinBigData or a hospital system to train/fine-tune an in-house detection model, or
   - Integrate an FDA-cleared commercial DICOM mammography CAD/AI partner via standard DICOM Web / REST interfaces.
