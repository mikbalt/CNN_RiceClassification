import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import requests
import os
from datetime import datetime
import json
from PIL import Image
import matplotlib.pyplot as plt

class RiceClassificationSystem:
    def __init__(self):
        self.model = None
        self.class_names = ['matang', 'mentah', 'setengah_matang']
        self.img_size = (224, 224)
        self.telegram_token = "7980398666:AAGHGy-0GA9piseTjAjTsBR52H9uMczY0Qc"
        self.telegram_chat_id = None  # Akan diisi nanti
        self.cap = None
        
    def get_telegram_chat_id(self):
        """Mendapatkan chat ID dari Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data['result']:
                    # Ambil chat_id dari pesan terakhir
                    chat_id = data['result'][-1]['message']['chat']['id']
                    print(f"Chat ID ditemukan: {chat_id}")
                    return str(chat_id)
                else:
                    print("Tidak ada pesan ditemukan.")
                    print("Silakan kirim pesan '/start' ke bot Anda terlebih dahulu di Telegram.")
                    return None
            else:
                print(f"Error getting updates: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting chat ID: {e}")
            return None
    
    def test_telegram_connection(self):
        """Test koneksi ke Telegram bot"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getMe"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data['ok']:
                    bot_info = data['result']
                    print(f"✅ Bot berhasil terkoneksi!")
                    print(f"Bot Name: {bot_info['first_name']}")
                    print(f"Bot Username: @{bot_info['username']}")
                    return True
                else:
                    print("❌ Bot tidak dapat diakses")
                    return False
            else:
                print(f"❌ Error connecting to bot: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing connection: {e}")
            return False
    
    def create_model(self):
        """Membuat model CNN untuk klasifikasi kematangan padi"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(3, activation='softmax')  # 3 kelas: matang, mentah, setengah_matang
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data_dir):
        """Mempersiapkan data untuk training"""
        # Data augmentation untuk menambah variasi dataset
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        # Generator untuk data training
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Generator untuk data validation
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train_model(self, data_dir, epochs=50):
        """Training model CNN"""
        print("Mempersiapkan data...")
        train_gen, val_gen = self.prepare_data(data_dir)
        
        print("Membuat model...")
        self.create_model()
        
        print("Mulai training...")
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // val_gen.batch_size,
            verbose=1
        )
        
        # Simpan model
        self.model.save('rice_classification_model.h5')
        print("Model berhasil disimpan sebagai 'rice_classification_model.h5'")
        
        return history
    
    def load_model(self, model_path='rice_classification_model.h5'):
        """Load model yang sudah di-training"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model berhasil dimuat!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocessing gambar sebelum prediksi"""
        # Resize gambar
        image = cv2.resize(image, self.img_size)
        # Normalisasi pixel values
        image = image.astype('float32') / 255.0
        # Tambah dimensi batch
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict_image(self, image):
        """Prediksi kematangan padi dari gambar"""
        if self.model is None:
            print("Model belum dimuat!")
            return None, None
        
        # Preprocessing
        processed_image = self.preprocess_image(image)
        
        # Prediksi
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_class = self.class_names[predicted_class_index]
        
        return predicted_class, confidence
    
    def send_telegram_notification(self, message, image_path=None):
        """Kirim notifikasi ke Telegram"""
        try:
            # Pastikan chat_id sudah ada
            if not self.telegram_chat_id:
                print("Chat ID belum diatur!")
                return False
            
            # Kirim pesan teks
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    print("✅ Pesan berhasil dikirim ke Telegram!")
                    
                    # Jika ada gambar, kirim juga gambar
                    if image_path and os.path.exists(image_path):
                        return self.send_telegram_photo(image_path)
                    return True
                else:
                    print(f"❌ Error dari Telegram API: {result}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending telegram notification: {e}")
            return False
    
    def send_telegram_photo(self, image_path):
        """Kirim foto ke Telegram"""
        try:
            url_photo = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data_photo = {'chat_id': self.telegram_chat_id}
                response = requests.post(url_photo, files=files, data=data_photo, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result['ok']:
                    print("✅ Foto berhasil dikirim ke Telegram!")
                    return True
                else:
                    print(f"❌ Error mengirim foto: {result}")
                    return False
            else:
                print(f"❌ HTTP Error saat mengirim foto: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending photo: {e}")
            return False
    
    def initialize_camera(self, source=0):
        """Inisialisasi kamera/CCTV"""
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print("Error: Tidak dapat membuka kamera!")
                return False
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def run_realtime_classification(self):
        """Jalankan sistem klasifikasi real-time"""
        if not self.initialize_camera():
            return
        
        if not self.load_model():
            print("Tidak dapat memuat model! Pastikan model sudah di-training.")
            return
        
        print("Sistem klasifikasi berjalan...")
        print("Tekan SPACE untuk capture dan klasifikasi")
        print("Tekan 'q' untuk keluar")
        
        capture_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error membaca frame dari kamera!")
                break
            
            # Tampilkan frame
            cv2.imshow('Rice Classification System - Press SPACE to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture dan klasifikasi saat SPACE ditekan
            if key == ord(' '):
                capture_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Simpan gambar
                image_filename = f"captured_rice_{timestamp}.jpg"
                cv2.imwrite(image_filename, frame)
                
                # Prediksi
                predicted_class, confidence = self.predict_image(frame)
                
                if predicted_class:
                    # Buat pesan notifikasi
                    message = f"""🌾 <b>Hasil Klasifikasi Kematangan Padi</b>

📅 Waktu: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🏷️ Klasifikasi: <b>{predicted_class.upper()}</b>
📊 Confidence: {confidence:.2%}
📸 Capture #{capture_count}

{self.get_recommendation(predicted_class)}"""
                    
                    # Kirim notifikasi ke Telegram
                    if self.telegram_chat_id:
                        self.send_telegram_notification(message, image_filename)
                    
                    # Tampilkan hasil di console
                    print(f"Hasil klasifikasi: {predicted_class} (Confidence: {confidence:.2%})")
                    
                    # Tampilkan hasil di frame
                    result_text = f"{predicted_class}: {confidence:.2%}"
                    cv2.putText(frame, result_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Classification Result', frame)
                    cv2.waitKey(2000)  # Tampilkan hasil selama 2 detik
            
            # Keluar jika 'q' ditekan
            elif key == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def get_recommendation(self, predicted_class):
        """Memberikan rekomendasi berdasarkan klasifikasi"""
        recommendations = {
            'matang': "✅ Padi sudah matang dan siap untuk dipanen!",
            'mentah': "❌ Padi masih belum matang. Tunggu beberapa minggu lagi.",
            'setengah_matang': "⚠️ Padi setengah matang. Pantau terus perkembangannya."
        }
        return recommendations.get(predicted_class, "")

# Fungsi untuk setup struktur folder dataset
def setup_dataset_structure():
    """Membuat struktur folder untuk dataset"""
    classes = ['matang', 'mentah', 'setengah_matang']
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    for class_name in classes:
        class_dir = os.path.join('dataset', class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Folder {class_dir} dibuat. Silakan masukkan 30 gambar untuk kelas {class_name}")

# Fungsi untuk test Telegram bot
def test_telegram_bot():
    """Test koneksi Telegram bot"""
    print("=== Test Telegram Bot ===")
    rice_system = RiceClassificationSystem()
    
    # Test koneksi bot
    print("1. Testing bot connection...")
    if rice_system.test_telegram_connection():
        print("\n2. Mencari Chat ID...")
        print("Jika belum pernah mengirim pesan ke bot, silakan:")
        print("1. Buka Telegram")
        print("2. Cari @Info_Padi_bot")
        print("3. Kirim pesan '/start' atau pesan apa saja")
        print("4. Kembali ke sini dan tekan Enter")
        
        input("Tekan Enter setelah mengirim pesan ke bot...")
        
        chat_id = rice_system.get_telegram_chat_id()
        if chat_id:
            rice_system.telegram_chat_id = chat_id
            
            # Test kirim pesan
            print("\n3. Testing send message...")
            test_message = "🤖 Test koneksi bot berhasil!\n\nBot siap digunakan untuk sistem klasifikasi padi."
            
            if rice_system.send_telegram_notification(test_message):
                print("✅ Semua test berhasil! Bot siap digunakan.")
                return True
            else:
                print("❌ Gagal mengirim pesan test.")
                return False
        else:
            print("❌ Tidak dapat mendapatkan Chat ID.")
            return False
    else:
        print("❌ Tidak dapat terhubung ke bot.")
        return False

# Contoh penggunaan
if __name__ == "__main__":
    print("=== Sistem Klasifikasi Kematangan Padi ===")
    print("0. Test Telegram Bot")
    print("1. Training Model")
    print("2. Jalankan Klasifikasi dari kamera")
    print("3. Test Single Image")
    
    choice = input("Pilih opsi (0/1/2/3): ")
    
    if choice == '0':
        # Test Telegram bot
        test_telegram_bot()
    
    elif choice == '1':
        # Training model
        setup_dataset_structure()
        print("Memulai training model...")
        print("Pastikan dataset sudah dimasukkan ke folder 'dataset' dengan struktur:")
        print("dataset/")
        print("├── matang/ (30 gambar)")
        print("├── mentah/ (30 gambar)")
        print("└── setengah_matang/ (30 gambar)")
        
        confirm = input("Apakah dataset sudah siap? (y/n): ")
        if confirm.lower() == 'y':
            rice_system = RiceClassificationSystem()
            history = rice_system.train_model('dataset', epochs=50)
            print("Training selesai!")
        else:
            print("Siapkan dataset terlebih dahulu.")
    
    elif choice == '2':
        # Jalankan klasifikasi real-time
        rice_system = RiceClassificationSystem()
        
        print("Pastikan model sudah di-training!")
        print("Setup Telegram bot...")
        
        # Test koneksi dan dapatkan chat ID
        if rice_system.test_telegram_connection():
            chat_id = rice_system.get_telegram_chat_id()
            if chat_id:
                rice_system.telegram_chat_id = chat_id
                print(f"Chat ID berhasil diatur: {chat_id}")
                rice_system.run_realtime_classification()
            else:
                print("Tidak dapat mendapatkan Chat ID. Jalankan tanpa notifikasi Telegram.")
                rice_system.run_realtime_classification()
        else:
            print("Bot tidak dapat terhubung. Jalankan tanpa notifikasi Telegram.")
            rice_system.run_realtime_classification()
    
    elif choice == '3':
        # Test single image
        rice_system = RiceClassificationSystem()
        image_path = input("Masukkan path gambar untuk di-test: ")

        if os.path.exists(image_path):
            if rice_system.load_model():
                image = cv2.imread(image_path)
                predicted_class, confidence = rice_system.predict_image(image)
                print(f"Hasil prediksi: {predicted_class} (Confidence: {confidence:.2%})")

                # Tambahan: Kirim ke Telegram
                if rice_system.test_telegram_connection():
                    chat_id = rice_system.get_telegram_chat_id()
                    if chat_id:
                        rice_system.telegram_chat_id = chat_id
                        message = f"""🌾 <b>Hasil Klasifikasi Kematangan Padi</b>

📅 Waktu: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🏷️ Klasifikasi: <b>{predicted_class.upper()}</b>
📊 Confidence: {confidence:.2%}

{self.get_recommendation(predicted_class)}"""
                        rice_system.send_telegram_notification(message, image_path)
                    else:
                        print("❌ Chat ID tidak ditemukan. Kirim '/start' ke bot terlebih dahulu.")
                else:
                    print("❌ Tidak bisa terhubung ke bot Telegram.")
            else:
                print("Model tidak ditemukan. Lakukan training terlebih dahulu.")
        else:
            print("File gambar tidak ditemukan!")
    
    else:
        print("Pilihan tidak valid!")