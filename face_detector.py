import cv2
import mediapipe as mp
import time
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

output_dir = "foto_wajah_cropped"
os.makedirs(output_dir, exist_ok=True)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    last_saved_time = 0
    save_interval = 5  # detik
    last_face_detected_time = 0
    alert_duration = 2  # detik, lama teks peringatan muncul

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca dari kamera.")
            break

        height, width, _ = frame.shape

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        face_detected = False

        if results.detections:
            face_detected = True
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

            current_time = time.time()
            if current_time - last_saved_time > save_interval:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * width)
                    y_min = int(bboxC.ymin * height)
                    w = int(bboxC.width * width)
                    h = int(bboxC.height * height)

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_min + w)
                    y_max = min(height, y_min + h)

                    cropped_face = frame[y_min:y_max, x_min:x_max]

                    filename = os.path.join(output_dir, f"wajah_crop_{int(current_time)}.jpg")
                    cv2.imwrite(filename, cropped_face)
                    print(f"Foto wajah crop tersimpan: {filename}")

                last_saved_time = current_time
                last_face_detected_time = current_time  # update waktu terakhir wajah terdeteksi

        # Tampilkan peringatan jika wajah terdeteksi dalam 2 detik terakhir
        if time.time() - last_face_detected_time < alert_duration:
            # Pilih font dan posisi teks
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Anda bukan ROBOT"
            org = (30, 50)
            font_scale = 1.5
            color = (0, 255, 0)  # hijau
            thickness = 3

            # Gambar teks di frame
            cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.namedWindow('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MediaPipe Face Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
