import cv2
import os
import time
from fer import FER

# Inisialisasi detektor wajah dan ekspresi
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_detector = FER()

# Buat folder output
output_dir = "output_faces"
os.makedirs(output_dir, exist_ok=True)

# Buka kamera
cap = cv2.VideoCapture(0)

last_saved_time = time.time()
save_interval = 5  # detik

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Tampilkan peringatan
        cv2.putText(frame, "Anda bukan ROBOT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 2, cv2.LINE_AA)

    for (x, y, w, h) in faces:
        # Kotak wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Ekstrak wajah
        face_crop = frame[y:y+h, x:x+w]

        # Deteksi ekspresi
        emotion_result = emotion_detector.detect_emotions(face_crop)
        if emotion_result:
            top_emotion = max(emotion_result[0]['emotions'], key=emotion_result[0]['emotions'].get)
            confidence = emotion_result[0]['emotions'][top_emotion]
            text_emotion = f"{top_emotion.capitalize()} ({confidence*100:.0f}%)"
            cv2.putText(frame, text_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Simpan tiap 5 detik
        current_time = time.time()
        if current_time - last_saved_time >= save_interval:
            filename = os.path.join(output_dir, f"face_{int(current_time)}.jpg")
            cv2.imwrite(filename, face_crop)
            print(f"[✅] Wajah tersimpan: {filename}")
            last_saved_time = current_time

    cv2.imshow("Deteksi Wajah dan Ekspresi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
