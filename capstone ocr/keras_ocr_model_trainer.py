import keras_ocr
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import utils
import config

class KerasOCRModelTrainer:
    def __init__(self, initial_weights=config.KERAS_OCR_INITIAL_WEIGHTS):
        self.recognizer = None
        self.initial_weights = initial_weights
        self.trained_model_path = os.path.join(
            config.SAVED_MODELS_DIR,
            config.KERAS_OCR_MODEL_CHECKPOINT_FILENAME
        )

    def _initialize_recognizer(self, alphabet):
        print(f"Menginisialisasi Keras-OCR Recognizer dengan alphabet kustom (panjang: {len(alphabet)}).")
        try:
            self.recognizer = keras_ocr.recognition.Recognizer(
                alphabet=alphabet,
                weights=self.initial_weights
            )
            self.recognizer.compile()
            print("Keras-OCR Recognizer berhasil diinisialisasi dan dikompilasi.")
        except Exception as e:
            print(f"Error membuat Keras-OCR Recognizer dengan alphabet kustom: {e}.")
            self.recognizer = None

    def train(self, images_train, labels_train):
        if not images_train or not labels_train:
            print("Data training (gambar atau label) kosong. Pelatihan dibatalkan.")
            return False

        temp_recognizer_for_alphabet = keras_ocr.pipeline.Pipeline().recognizer
        default_keras_ocr_alphabet = temp_recognizer_for_alphabet.alphabet

        custom_alphabet = utils.create_custom_alphabet(
            labels_train,
            default_keras_ocr_alphabet
        )
        if not custom_alphabet:
            print("Gagal membuat alphabet kustom yang valid. Pelatihan dibatalkan.")
            return False

        self._initialize_recognizer(custom_alphabet)
        if self.recognizer is None:
            print("Keras-OCR Recognizer tidak berhasil diinisialisasi. Pelatihan dibatalkan.")
            return False

        epochs = config.KERAS_OCR_TRAINING_EPOCHS
        batch_size = config.KERAS_OCR_TRAINING_BATCH_SIZE
        learning_rate = config.KERAS_OCR_TRAINING_LEARNING_RATE

        current_optimizer = self.recognizer.model.optimizer
        if hasattr(current_optimizer, 'learning_rate'):
            current_optimizer.learning_rate.assign(learning_rate)

        print(f"Learning rate Keras-OCR Recognizer diatur/dikonfirmasi ke: {self.recognizer.model.optimizer.learning_rate.numpy()}")

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            filepath=self.trained_model_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        callbacks_list = [early_stopping, model_checkpoint]

        print(f"\nMemulai fine-tuning Keras-OCR Recognizer dengan {epochs} epoch...")
        try:
            history = self.recognizer.fit(
                images_true=images_train,
                labels_true=labels_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
            print("Fine-tuning Keras-OCR Recognizer selesai.")

            if os.path.exists(self.trained_model_path) and model_checkpoint.save_best_only:
                print(f"Memuat bobot Keras-OCR Recognizer terbaik dari: {self.trained_model_path}")
                self.recognizer.model.load_weights(self.trained_model_path)
                print("Bobot terbaik berhasil dimuat ke Keras-OCR Recognizer.")

            print(f"Model Keras-OCR Recognizer yang di-fine-tune (atau bobot terbaiknya) disimpan di: {self.trained_model_path}")
            return True
        except Exception as e:
            print(f"Error selama training Keras-OCR Recognizer: {e}")
            return False

    def get_trained_recognizer(self):
        if self.recognizer and os.path.exists(self.trained_model_path):
            print(f"Memastikan bobot terbaik dimuat dari {self.trained_model_path} untuk recognizer.")
            try:
                self.recognizer.model.load_weights(self.trained_model_path)
            except Exception as e:
                print(f"Gagal memuat bobot ke recognizer yang ada: {e}. Mungkin perlu reinisialisasi dengan alphabet yang benar.")
                return None
        elif not self.recognizer and os.path.exists(self.trained_model_path):
            print("Recognizer belum ada, tapi file bobot ditemukan. Perlu alphabet untuk memuat.")
            return None
        return self.recognizer