import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

# Path ke model yang telah diekspor dan label map
PATH_TO_SAVED_MODEL = 'D:\Deploy2\inference_graph\saved_model'  # Ganti dengan path ke SavedModel Anda
PATH_TO_LABELS = 'D:\Deploy2\Vehicle_label_map.pbtxt'  # Ganti dengan path ke label map

# Memuat model yang telah diekspor dalam format SavedModel
print("Memuat model...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model berhasil dimuat!")

# Memuat label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Membuka video
cap = cv2.VideoCapture('D:\Deploy2\Video_Sample.mp4')  # Ganti dengan path ke video Anda

# Loop untuk membaca setiap frame dari video
while cap.isOpened():
    ret, image_np = cap.read()
    if not ret:
        print("Video selesai atau tidak dapat dibaca.")
        break

    # Konversi frame menjadi tensor untuk model
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Deteksi objek
    detections = detect_fn(input_tensor)

    # Memproses hasil deteksi
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Konversi kelas ke integer
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualisasi kotak deteksi dan label pada frame
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Menampilkan frame yang telah diproses
    cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))

    # Tekan 'q' untuk keluar dari video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Tutup video dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
