import os
from PIL import Image
import matplotlib.pyplot as plt
import keras_ocr

import config
from donut_model_handler import DonutModelHandler
from text_detector_cropper import TextDetectorCropper
from keras_ocr_model_trainer import KerasOCRModelTrainer

def run_ocr_pipeline():
    print("Memulai Alur Kerja OCR dengan Pseudo-Labeling...")

    print("\n--- Langkah 1: Inisialisasi Komponen ---")
    donut_handler = DonutModelHandler()
    if donut_handler.model is None:
        print("Gagal memuat model Donut. Proses dihentikan.")
        return

    text_detector = TextDetectorCropper()
    keras_ocr_trainer = KerasOCRModelTrainer()

    print("\n--- Langkah 2: Persiapan Dataset dengan Pseudo-Label dari Donut ---")
    images_for_training = []
    pseudo_labels_for_training = []

    if not os.path.exists(config.FULL_LABEL_IMAGE_DIR):
        print(f"Error: Direktori gambar label utuh tidak ditemukan: {config.FULL_LABEL_IMAGE_DIR}")
        print("Pastikan path di config.py benar dan folder berisi gambar.")
        return

    all_full_image_paths = [os.path.join(config.FULL_LABEL_IMAGE_DIR, f)
                            for f in os.listdir(config.FULL_LABEL_IMAGE_DIR)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not all_full_image_paths:
        print(f"Tidak ada gambar yang ditemukan di {config.FULL_LABEL_IMAGE_DIR}")
        return

    num_images_to_process = len(all_full_image_paths)
    if config.MAX_FULL_IMAGES_TO_PROCESS is not None:
        num_images_to_process = min(len(all_full_image_paths), config.MAX_FULL_IMAGES_TO_PROCESS)
        print(f"Akan memproses {num_images_to_process} gambar (dibatasi oleh MAX_FULL_IMAGES_TO_PROCESS).")

    for i, full_image_path in enumerate(all_full_image_paths[:num_images_to_process]):
        print(f"\nMemproses gambar utuh: {os.path.basename(full_image_path)} ({i+1}/{num_images_to_process})")

        cropped_results = text_detector.get_cropped_text_images(full_image_path)

        for cropped_image_np, _ in cropped_results:
            try:
                cropped_image_pil = Image.fromarray(cropped_image_np)
                pseudo_label = donut_handler.generate_pseudo_label(cropped_image_pil)

                if "ERROR" not in pseudo_label and pseudo_label.strip():
                    images_for_training.append(cropped_image_np)
                    pseudo_labels_for_training.append(pseudo_label)
                elif not pseudo_label.strip():
                    print(f"    Peringatan: Pseudo-label kosong dari Donut untuk sebuah crop, dilewati.")
                else:
                    print(f"    Peringatan: Gagal mendapatkan pseudo-label valid ('{pseudo_label}') dari Donut, dilewati.")
            except Exception as e_crop:
                print(f"    Error saat memproses crop dari {os.path.basename(full_image_path)}: {e_crop}")

    if not images_for_training:
        print("\nTidak ada data training (potongan gambar dengan pseudo-label) yang berhasil dibuat.")
        print("Proses dihentikan. Periksa output dari Donut atau kualitas gambar input.")
        return

    print(f"\nBerhasil membuat {len(images_for_training)} sampel data training dengan pseudo-label.")

    if images_for_training and os.environ.get('DISPLAY'):
        try:
            print("\nContoh data training (gambar crop dengan pseudo-label Donut):")
            plt.figure(figsize=(15, 7))
            num_examples_to_show = min(10, len(images_for_training))
            example_indices = random.sample(range(len(images_for_training)), num_examples_to_show)
            for plot_i, data_i in enumerate(example_indices):
                plt.subplot(2, 5, plot_i + 1)
                plt.imshow(images_for_training[data_i])
                plt.title(f"Donut: {pseudo_labels_for_training[data_i][:25]}...", fontsize=8)
                plt.axis('off')
            plt.suptitle("Contoh Potongan Gambar Teks dengan Pseudo-Label dari Donut")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        except Exception as e_plot:
            print(f"Gagal menampilkan plot contoh: {e_plot}. Mungkin karena tidak ada environment GUI.")

    print("\n--- Langkah 3: Training Keras-OCR Recognizer ---")
    training_successful = keras_ocr_trainer.train(images_for_training, pseudo_labels_for_training)

    if not training_successful:
        print("Training Keras-OCR Recognizer tidak berhasil. Proses dihentikan.")
        return

    print("\n--- Langkah 4: Contoh Penggunaan Pipeline Keras-OCR yang Sudah Di-fine-tune ---")
    trained_recognizer = keras_ocr_trainer.get_trained_recognizer()

    if trained_recognizer:
        final_keras_ocr_pipeline = keras_ocr.pipeline.Pipeline(
            detector=text_detector.keras_ocr_pipeline.detector,
            recognizer=trained_recognizer
        )
        print("Pipeline Keras-OCR final dengan recognizer terlatih siap digunakan.")

        if all_full_image_paths:
            test_image_path_demo = all_full_image_paths[0]
            print(f"\nMelakukan prediksi pada gambar demo: {os.path.basename(test_image_path_demo)}")
            try:
                test_image_np_demo = keras_ocr.tools.read(test_image_path_demo)
                prediction_groups = final_keras_ocr_pipeline.recognize([test_image_np_demo])

                print("Hasil Prediksi:")
                for text, box in prediction_groups[0]:
                    print(f"  Teks: '{text}', Box: {box.round().astype(int).tolist()}")

                if os.environ.get('DISPLAY'):
                    try:
                        plt.figure(figsize=(10,10))
                        keras_ocr.tools.drawAnnotations(image=test_image_np_demo, predictions=prediction_groups[0], ax=plt.gca())
                        plt.title(f"Prediksi pada {os.path.basename(test_image_path_demo)} (Keras-OCR fine-tuned)")
                        plt.show()
                    except Exception as e_plot_final:
                        print(f"Gagal menampilkan plot prediksi: {e_plot_final}")

            except Exception as e_pred:
                print(f"Error saat melakukan prediksi demo: {e_pred}")
        else:
            print("Tidak ada gambar untuk prediksi demo.")
    else:
        print("Tidak dapat mengambil Keras-OCR Recognizer yang sudah dilatih.")

    print("\nAlur Kerja OCR dengan Pseudo-Labeling Selesai.")

if __name__ == '__main__':
    run_ocr_pipeline()
