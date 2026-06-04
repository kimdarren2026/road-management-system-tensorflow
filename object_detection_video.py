import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import time
from datetime import datetime
import os

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Video settings
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
VIDEO_FPS = 120

# Output settings
OUTPUT_DIR = "output_videos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_class_names(filename):
    class_names = []
    try:
        with open(filename, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return None

def load_model(model_path):
    try:
        # Load saved model
        detect_fn = tf.saved_model.load(model_path)
        return detect_fn
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_objects(frame, detect_fn, class_names):
    # Initialize metrics for this frame in the desired order
    metrics = {
        'TP': 0,  # True Positives
        'FP': 0,  # False Positives
        'TN': 0,  # True Negatives
        'FN': 0   # False Negatives
    }
    
    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    
    # Run inference
    start_time = time.time()
    detections = detect_fn(input_tensor)
    inference_time = time.time() - start_time
    
    # Process detections
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    
    # Debug print
    print("\nAll detections before filtering:")
    for i in range(min(5, len(scores))):
        print(f"Detection {i}: class={classes[i]} ({class_names[classes[i]-1] if 0 <= classes[i]-1 < len(class_names) else 'unknown'}), score={scores[i]:.2f}")
    
    height, width = frame.shape[:2]
    
    # Process detections
    detection_threshold = 0.4
    for i in range(len(scores)):
        if scores[i] > detection_threshold:
            # Get coordinates
            box = boxes[i]
            y1, x1, y2, x2 = box
            x1 = int(x1 * width)
            x2 = int(x2 * width)
            y1 = int(y1 * height)
            y2 = int(y2 * height)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add label
            class_idx = classes[i] - 1  # TensorFlow model outputs 1-based indices
            if 0 <= class_idx < len(class_names):
                label = f"{class_names[class_idx]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
            
            # Update metrics
            if scores[i] > 0.7:  # High confidence threshold for True Positive
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        else:
            # Count as True Negative if score is very low
            if scores[i] < 0.2:
                metrics['TN'] += 1
            else:
                metrics['FN'] += 1
    
    # Draw metrics on frame in the specified order
    metrics_text = f"TP: {metrics['TP']} | FP: {metrics['FP']} | TN: {metrics['TN']} | FN: {metrics['FN']}"
    cv2.putText(frame, metrics_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Format output string like YOLO
    output_str = f"{width}x{height}"
    object_counts = {}
    for i in range(len(scores)):
        if scores[i] > detection_threshold:
            class_idx = classes[i] - 1  # TensorFlow model outputs 1-based indices
            if 0 <= class_idx < len(class_names):
                class_name = class_names[class_idx].strip()  # Remove any whitespace
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
    for obj_name, count in sorted(object_counts.items()):  # Sort by class name
        output_str += f" {count} {obj_name},"
    output_str = output_str.rstrip(',')
    output_str += f", Done. ({inference_time:.4f}s)"
    
    print(output_str)
    
    return frame, metrics

def main():
    print("\nInitializing object detection with GPU acceleration...")
    print("TensorFlow version:", tf.__version__)
    
    print("\nLoading model... This might take a few minutes.")
    # Load model
    model_path = "inference_graph_msdaren\saved_model" #GANTI MODEL E WONG
    
    detect_fn = load_model(model_path)
    if detect_fn is None:
        return
    
    # Load class names
    class_names = load_class_names("predefined_classes.txt")
    if class_names is None:
        return
    
    print("Model loaded successfully! Opening video...")

    # Open video capture
    # cap = cv2.VideoCapture("TES-VIDEO.mkv") 
    cap = cv2.VideoCapture(0)  #GANTI VIDEO, KAMERA. KET : 0 = KAMERA, 1 = EKSTERNAL
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Original video properties - Width: {original_width}, Height: {original_height}, FPS: {original_fps}")
    
    # Use original video properties instead of preset values
    actual_width = original_width
    actual_height = original_height
    actual_fps = original_fps
    
    # Create video writer with original dimensions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"test_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (actual_width, actual_height))
    print(f"Saving output video to: {output_path}")
    
    cv2.namedWindow("Object Detection", cv2.WINDOW_AUTOSIZE)
    
    frame_count = 0
    start_time = time.time()
    processing_times = []
    
    # Initialize total metrics in the desired order
    total_metrics = {
        'TP': 0,
        'FP': 0,
        'TN': 0,
        'FN': 0
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_start_time = time.time()
        
        # Convert BGR to RGB (TensorFlow expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        processed_frame, metrics = detect_objects(frame_rgb, detect_fn, class_names)
        
        frame_time = time.time() - frame_start_time
        processing_times.append(frame_time)
        
        # Convert back to BGR for display and saving
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        
        # Update total metrics
        total_metrics['TP'] += metrics['TP']
        total_metrics['FP'] += metrics['FP']
        total_metrics['TN'] += metrics['TN']
        total_metrics['FN'] += metrics['FN']
        
        # Write frame to output video
        out.write(processed_frame)
        
        # Calculate and display FPS and inference time
        current_time = time.time()
        fps = 1.0 / (current_time - frame_start_time)
        inference_time = (current_time - frame_start_time) * 1000  # Convert to ms
        
        # Display metrics on frame
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Inference: {inference_time:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display result
        cv2.imshow("Object Detection", processed_frame)
        
        # Break on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    average_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_process_time = sum(processing_times) / len(processing_times) * 1000 if processing_times else 0  # Convert to ms
    print(f"\nProcessing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Average processing time per frame: {avg_process_time:.1f}ms")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Output video saved to: {output_path}")
    
    # Calculate and display final metrics in the specified order
    print("\nFinal Detection Metrics:")
    print(f"Total True Positives (TP): {total_metrics['TP']}")
    print(f"Total False Positives (FP): {total_metrics['FP']}")
    print(f"Total True Negatives (TN): {total_metrics['TN']}")
    print(f"Total False Negatives (FN): {total_metrics['FN']}")
    
    # Cleanup
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()