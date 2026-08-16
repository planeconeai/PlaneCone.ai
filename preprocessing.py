import io
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("mammo-preprocessing")

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom or apply_voi_lut not available — fallback image parser will be used")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def crop_breast_background(arr: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Crops empty black background pixels around breast tissue to preserve contrast
    and focus on mammographic parenchyma.
    """
    mask = arr > threshold
    if not np.any(mask):
        return arr
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Ensure valid bounding box
    if ymax > ymin and xmax > xmin:
        return arr[ymin:ymax + 1, xmin:xmax + 1]
    return arr

def parse_dicom_bytes(file_bytes: bytes, filename: str = "image.dcm") -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Parses raw DICOM byte array following official Mammo-CLIP & DICOM standards:
    1. Reads DICOM header & dataset.
    2. Applies VOI LUT / Windowing if present (Window Center/Width).
    3. Handles Rescale Slope & Intercept.
    4. Handles Photometric Interpretation (MONOCHROME1 vs MONOCHROME2).
    5. Crops empty background black padding around breast tissue.
    6. Normalizes pixel values to range [0.0, 1.0].
    """
    metadata = {
        "filename": filename,
        "modality": "MG",
        "view_position": "UNKNOWN",
        "study_instance_uid": None,
        "series_instance_uid": None,
        "sop_instance_uid": None,
        "photometric_interpretation": "UNKNOWN",
        "has_voi_lut": False,
    }

    image_array = None

    if PYDICOM_AVAILABLE:
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
            metadata["modality"] = getattr(ds, "Modality", "MG") or "MG"
            metadata["view_position"] = (getattr(ds, "ViewPosition", "") or "").upper() or "UNKNOWN"
            metadata["study_instance_uid"] = getattr(ds, "StudyInstanceUID", None)
            metadata["series_instance_uid"] = getattr(ds, "SeriesInstanceUID", None)
            metadata["sop_instance_uid"] = getattr(ds, "SOPInstanceUID", None)
            
            photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
            metadata["photometric_interpretation"] = photo_interp

            if hasattr(ds, "pixel_array"):
                # Step 1: Apply VOI LUT (Windowing) if available in DICOM header
                try:
                    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
                    metadata["has_voi_lut"] = True
                except Exception:
                    arr = ds.pixel_array.astype(np.float32)

                # Step 2: Rescale Slope and Intercept
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                if slope != 1.0 or intercept != 0.0:
                    arr = arr * slope + intercept

                # Step 3: Photometric Interpretation (MONOCHROME1 invert)
                if photo_interp == "MONOCHROME1":
                    arr = np.max(arr) - arr

                # Step 4: Min-Max normalize array to [0.0, 1.0]
                min_val = np.min(arr)
                max_val = np.max(arr)
                if max_val > min_val:
                    arr = (arr - min_val) / (max_val - min_val)
                else:
                    arr = np.zeros_like(arr)

                # Step 5: Crop background black padding around breast tissue
                image_array = crop_breast_background(arr)
        except Exception as e:
            logger.warning(f"pydicom processing failed for {filename}: {e}")

    # Fallback parser if pydicom unavailable or failed
    if image_array is None and PIL_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            image_array = crop_breast_background(arr)
        except Exception as e:
            logger.warning(f"PIL fallback parser failed for {filename}: {e}")

    # Infer view position from filename if missing from DICOM header
    if metadata["view_position"] == "UNKNOWN":
        upper_fn = filename.upper()
        if "RCC" in upper_fn or "R_CC" in upper_fn or "R-CC" in upper_fn:
            metadata["view_position"] = "R-CC"
        elif "LCC" in upper_fn or "L_CC" in upper_fn or "L-CC" in upper_fn:
            metadata["view_position"] = "L-CC"
        elif "RMLO" in upper_fn or "R_MLO" in upper_fn or "R-MLO" in upper_fn:
            metadata["view_position"] = "R-MLO"
        elif "LMLO" in upper_fn or "L_MLO" in upper_fn or "L-MLO" in upper_fn:
            metadata["view_position"] = "L-MLO"
        elif "CC" in upper_fn:
            metadata["view_position"] = "CC"
        elif "MLO" in upper_fn:
            metadata["view_position"] = "MLO"

    return image_array, metadata

def prepare_image_tensor(image_array: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resizes image_array preserving aspect ratio with padding to target_size (224x224),
    replicates to 3 RGB channels, and applies CLIP ImageNet normalization.
    """
    if PIL_AVAILABLE and image_array is not None:
        img = Image.fromarray((image_array * 255).astype(np.uint8))
        
        # Aspect-ratio preserving resize with black padding
        img.thumbnail(target_size, Image.BILINEAR)
        padded_img = Image.new("L", target_size, 0)
        paste_pos = ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2)
        padded_img.paste(img, paste_pos)

        arr = np.array(padded_img, dtype=np.float32) / 255.0
    elif image_array is not None:
        arr = image_array
    else:
        arr = np.zeros(target_size, dtype=np.float32)

    # 3-channel RGB
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        rgb = arr

    # CLIP Normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    norm_rgb = (rgb - mean) / std
    chw = np.transpose(norm_rgb, (2, 0, 1))
    return chw
