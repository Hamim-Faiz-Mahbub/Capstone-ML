import pytesseract
from PIL import Image
import os

class TesseractOCRHandler:
    def __init__(self, language='eng'):
        self.language = language
        print(f"Tesseract OCR Handler diinisialisasi dengan bahasa: {self.language}")
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Versi Tesseract yang terdeteksi: {version}")
        except Exception as e:
            print(f"Peringatan: Tidak dapat memverifikasi versi Tesseract. Pastikan Tesseract terinstal. Error: {e}")
            print("Jika di Colab, coba jalankan: !sudo apt install tesseract-ocr")

    def generate_pseudo_label(self, image_crop_pil: Image.Image) -> str:
        if not isinstance(image_crop_pil, Image.Image):
            print("Error: Input bukan objek PIL Image yang valid.")
            return "ERROR_INVALID_IMAGE_INPUT"

        try:
            text = pytesseract.image_to_string(image_crop_pil, lang=self.language)
            cleaned_text = text.strip()
            if not cleaned_text:
                return ""
            return cleaned_text
        except pytesseract.TesseractNotFoundError:
            print("CRITICAL ERROR: Tesseract executable tidak ditemukan. Pastikan Tesseract terinstal.")
            print("Jika di Colab, jalankan: !sudo apt install tesseract-ocr")
            return "ERROR_TESSERACT_NOT_FOUND"
        except Exception as e:
            print(f"Error saat inferensi Tesseract: {e}")
            return "ERROR_TESSERACT_INFERENCE"
