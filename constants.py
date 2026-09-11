"""
Mammo-AI Service Constants
Defines configuration for the MammoCLIP model and inference settings
"""

import os

# ============ SERVER CONFIGURATION ============
SERVER = {
    'HOST': os.getenv('MAMMO_AI_HOST', '0.0.0.0'),
    'PORT': int(os.getenv('MAMMO_AI_PORT', 8000)),
    'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
    'DEBUG_MODE': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
    'WORKERS': int(os.getenv('MAMMO_AI_WORKERS', 1)),
    'RELOAD': os.getenv('MAMMO_AI_RELOAD', 'false').lower() == 'true',
}

# ============ MODEL CONFIGURATION ============
MODEL = {
    'NAME': os.getenv('MODEL_NAME', 'mammo-clip-resnet50'),
    'VERSION': os.getenv('MODEL_VERSION', '1.0.0'),
    'CHECKPOINT_PATH': os.getenv('MODEL_CHECKPOINT_PATH', './models/mammo_clip.pth'),
    'DEVICE': os.getenv('MODEL_DEVICE', 'cpu'),  # 'cpu' or 'cuda'
    'PRECISION': os.getenv('MODEL_PRECISION', 'float32'),  # 'float32' or 'float16'
    'BATCH_SIZE': int(os.getenv('MODEL_BATCH_SIZE', 8)),
    'MAX_BATCH_SIZE': int(os.getenv('MODEL_MAX_BATCH_SIZE', 32)),
    'INPUT_SIZE': int(os.getenv('MODEL_INPUT_SIZE', 224)),
    'TIMEOUT_SECONDS': int(os.getenv('MODEL_TIMEOUT_SECONDS', 30)),
}

# ============ INFERENCE SETTINGS ============
INFERENCE = {
    'CONFIDENCE_THRESHOLD': float(os.getenv('INFERENCE_CONFIDENCE_THRESHOLD', 0.85)),
    'CONFIDENCE_RANGE': float(os.getenv('INFERENCE_CONFIDENCE_RANGE', 0.14)),
    'NUM_CLASSES': int(os.getenv('INFERENCE_NUM_CLASSES', 3)),
    'OUTPUT_FORMAT': os.getenv('INFERENCE_OUTPUT_FORMAT', 'json'),  # 'json' or 'dict'
}

# ============ PREPROCESSING ============
PREPROCESSING = {
    'NORMALIZE_MEAN': [
        float(x) for x in os.getenv('PREPROCESS_MEAN', '0.485,0.456,0.406').split(',')
    ],
    'NORMALIZE_STD': [
        float(x) for x in os.getenv('PREPROCESS_STD', '0.229,0.224,0.225').split(',')
    ],
    'RESIZE_METHOD': os.getenv('RESIZE_METHOD', 'bilinear'),
    'CONVERT_RGB': os.getenv('CONVERT_RGB', 'true').lower() == 'true',
    'CLIP_PIXEL_RANGE': (
        int(os.getenv('PIXEL_MIN', 0)),
        int(os.getenv('PIXEL_MAX', 255)),
    ),
}

# ============ POSTPROCESSING ============
POSTPROCESSING = {
    'APPLY_SOFTMAX': os.getenv('APPLY_SOFTMAX', 'true').lower() == 'true',
    'INCLUDE_CONFIDENCE': os.getenv('INCLUDE_CONFIDENCE', 'true').lower() == 'true',
    'INCLUDE_TIMING': os.getenv('INCLUDE_TIMING', 'true').lower() == 'true',
    'ROUND_PRECISION': int(os.getenv('ROUND_PRECISION', 4)),
}

# ============ CLASS DEFINITIONS ============
CLASSES = {
    0: 'Normal',
    1: 'Benign',
    2: 'Malignant',
}

DISEASE_CATEGORIES = {
    'Normal': ['No abnormality detected'],
    'Benign': ['Fibroadenoma', 'Cyst', 'Calcification', 'Fat necrosis'],
    'Malignant': [
        'Invasive Ductal Carcinoma',
        'Lobular Carcinoma',
        'Ductal Carcinoma In Situ',
    ],
}

# ============ CACHING ============
CACHE = {
    'ENABLED': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
    'TTL_SECONDS': int(os.getenv('CACHE_TTL_SECONDS', 3600)),
    'MAX_SIZE_MB': int(os.getenv('CACHE_MAX_SIZE_MB', 500)),
    'REDIS_URL': os.getenv('REDIS_URL', None),  # None = in-memory cache
}

# ============ LOGGING ============
LOGGING = {
    'LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    'FORMAT': os.getenv('LOG_FORMAT', 'json'),
    'FILE': os.getenv('LOG_FILE', './logs/mammo_ai.log'),
    'MAX_BYTES': int(os.getenv('LOG_MAX_BYTES', 10485760)),  # 10MB
    'BACKUP_COUNT': int(os.getenv('LOG_BACKUP_COUNT', 5)),
    'LOG_PREDICTIONS': os.getenv('LOG_PREDICTIONS', 'true').lower() == 'true',
    'LOG_PERFORMANCE': os.getenv('LOG_PERFORMANCE', 'true').lower() == 'true',
}

# ============ CORS ============
CORS = {
    'ENABLED': os.getenv('CORS_ENABLED', 'true').lower() == 'true',
    'ALLOW_ORIGINS': os.getenv('CORS_ALLOW_ORIGINS', '*').split(','),
    'ALLOW_CREDENTIALS': os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true',
    'ALLOW_METHODS': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    'ALLOW_HEADERS': ['Content-Type', 'Authorization'],
    'MAX_AGE': int(os.getenv('CORS_MAX_AGE', 3600)),
}

# ============ AUTHENTICATION ============
AUTHENTICATION = {
    'ENABLED': os.getenv('AUTH_ENABLED', 'false').lower() == 'true',
    'API_KEY': os.getenv('API_KEY', None),
    'JWT_SECRET': os.getenv('JWT_SECRET', None),
    'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', 'HS256'),
}

# ============ PERFORMANCE ============
PERFORMANCE = {
    'ENABLE_PROFILING': os.getenv('ENABLE_PROFILING', 'false').lower() == 'true',
    'THREAD_POOL_SIZE': int(os.getenv('THREAD_POOL_SIZE', 4)),
    'QUEUE_SIZE': int(os.getenv('QUEUE_SIZE', 100)),
    'ENABLE_ASYNC': os.getenv('ENABLE_ASYNC', 'true').lower() == 'true',
}

# ============ MONITORING & HEALTH ============
MONITORING = {
    'ENABLED': os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
    'METRICS_ENABLED': os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
    'HEALTH_CHECK_INTERVAL_SECONDS': int(
        os.getenv('HEALTH_CHECK_INTERVAL_SECONDS', 30)
    ),
}

# ============ TESTING ============
TESTING = {
    'ENABLED': os.getenv('TESTING_ENABLED', 'false').lower() == 'true',
    'TEST_MODE': os.getenv('TEST_MODE', 'false').lower() == 'true',
    'MOCK_PREDICTIONS': os.getenv('MOCK_PREDICTIONS', 'false').lower() == 'true',
}
