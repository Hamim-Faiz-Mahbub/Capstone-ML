import os

BASE_PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = '/content/drive/MyDrive/ocr_ocran/Dataset_capstone_ocr'
FULL_LABEL_IMAGE_DIR = DATA_PATH
SAVED_MODELS_DIR = '/content/drive/MyDrive/Colab Notebooks/ocr Capstone/'
KERAS_OCR_MODEL_CHECKPOINT_FILENAME = 'best_keras_ocr_recognizer_tesseract_sourced.h5'
TESSERACT_LANGUAGE = 'eng'
KERAS_OCR_INITIAL_WEIGHTS = 'craft'
MAX_FULL_IMAGES_TO_PROCESS = None
KERAS_OCR_TRAINING_EPOCHS = 30
KERAS_OCR_TRAINING_BATCH_SIZE = 8
KERAS_OCR_TRAINING_LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 7

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
