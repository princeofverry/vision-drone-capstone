import numpy as np
import tensorflow as tf
import cv2
from picamera2 import Picamera2
import time

# Inisialisasi kamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 640)}))
picam2.start()
time.sleep(2)  # Waktu untuk kamera menyesuaikan

# Load model TFLite
model_path = 'tflite-model/best_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['fire', 'smoke']
threshold = 0.2  # Confidence threshold

# Untuk hitung FPS
prev_time = time.time()

# Loop real-time
while True:
    # Hitung waktu mulai frame
    start_time = time.time()

    # Capture frame dari kamera
    frame = picam2.capture_array()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    
    # Preprocessing: normalize dan expand dims
    input_data = np.expand_dims(image_resized / 255.0, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-processing
    for i in range(output_data.shape[2]):
        detection = output_data[0, :, i]
        bbox = detection[0:4]
        confidence = detection[4]

        if confidence > threshold:
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_name = class_names[class_id]

            x_center, y_center, width, height = bbox
            x1 = int((x_center - width / 2) * 640)
            y1 = int((y_center - height / 2) * 640)
            x2 = int((x_center + width / 2) * 640)
            y2 = int((y_center + height / 2) * 640)

            # Gambar kotak dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Hitung dan tampilkan FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Real-time Detection", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cv2.destroyAllWindows()
picam2.stop()
