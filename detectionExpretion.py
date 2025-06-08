from fer import FER  # library untuk ekspresi wajah
import cv2
import mediapipe as mp
import time
import os


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detector_exp = FER(mtcnn=True)  # detektor ekspresi wajah

cap = cv2.VideoCapture(0)

output_dir = "foto_wajah_cropped"
os.makedirs(output_dir, exist_ok=True)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    last_saved_time = 0
    save_interval = 5
    last_face_detected_time = 0
    alert_duration = 2

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
        ekspresi_text = ""

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
                last_face_detected_time = current_time

            # Prediksi ekspresi wajah dengan FER (gunakan frame asli)
            try:
                ekspresi = detector_exp.detect_emotions(frame)
                if ekspresi:
                    # Ambil wajah pertama
                    face_emo = ekspresi[0]
                    emosi = face_emo['emotions']
                    # Ambil ekspresi dengan confidence tertinggi
                    max_emosi = max(emosi, key=emosi.get)
                    if max_emosi == "happy":
                        ekspresi_text = "Senang"
                    elif max_emosi == "sad":
                        ekspresi_text = "Sedih"
                    else:
                        ekspresi_text = ""
            except Exception as e:
                print("Error deteksi ekspresi:", e)

        # Tampilkan peringatan dan ekspresi jika ada
        if time.time() - last_face_detected_time < alert_duration and face_detected:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_alert = "Anda bukan ROBOT"
            org_alert = (200, 50)
            font_scale = 1
            color = (0, 255, 0)
            thickness = 1
            cv2.putText(image, text_alert, org_alert, font, font_scale, color, thickness, cv2.LINE_AA)

            if ekspresi_text:
                org_exp = (30, 100)
                color_exp = (0, 255, 255)
                cv2.putText(image, ekspresi_text, org_exp, font, font_scale, color_exp, thickness, cv2.LINE_AA)

        cv2.namedWindow('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('MediaPipe Face Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
