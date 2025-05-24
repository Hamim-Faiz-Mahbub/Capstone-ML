import keras_ocr
from PIL import Image
import numpy as np
import os
import config 

class TextDetectorCropper:
    def __init__(self):
        print("Menginisialisasi Keras-OCR pipeline (untuk detector CRAFT)...")
        self.keras_ocr_pipeline = keras_ocr.pipeline.Pipeline()
        print("Keras-OCR pipeline berhasil diinisialisasi.")

    def get_cropped_text_images(self, full_image_path):
        """
        Mendeteksi teks pada gambar utuh, memotongnya, dan mengembalikan list gambar potongan.
        Mengembalikan list tuple (cropped_image_numpy, original_box).
        """
        cropped_images_with_boxes = []
        try:
            full_image_np = keras_ocr.tools.read(full_image_path)
            if full_image_np is None:
                print(f"  Gagal membaca gambar: {full_image_path}")
                return []

            detection_results = self.keras_ocr_pipeline.detector.detect(images=[full_image_np])
            boxes = detection_results[0]
            print(f"  Terdeteksi {len(boxes)} kotak teks di {os.path.basename(full_image_path)}.")

            for box_idx, box in enumerate(boxes):
                cropped_text_image_np = keras_ocr.tools.warpBox(
                    image=full_image_np,
                    box=box.astype('float32') 
                )

                if cropped_text_image_np is not None and \
                   cropped_text_image_np.size > 0 and \
                   cropped_text_image_np.shape[0] >= 5 and \
                   cropped_text_image_np.shape[1] >= 5:
                    cropped_images_with_boxes.append((cropped_text_image_np, box))
                else:
                    print(f"    Kotak {box_idx+1}: Hasil crop tidak valid atau terlalu kecil, dilewati.")

            return cropped_images_with_boxes
        except Exception as e:
            print(f"  Error saat deteksi/cropping gambar {os.path.basename(full_image_path)}: {e}")
            return []