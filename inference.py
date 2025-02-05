from collections import deque
from scipy import stats

import tensorflow as tf
import numpy as np
import argparse
import cv2


def preprocess_frame(frame, img_width, img_height, normalization=True):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_width, img_height))
    if normalization:
        frame = frame.astype("float32") / 127.5 - 1
    else:
        frame = frame.astype("float32")
    frame = np.expand_dims(frame, axis=0)
    return frame


def main(args):

    # Load TFLite model
    model_path = f"results/models/{args.model_name}.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_height, img_width = input_details[0]['shape'][1:3]

    # Define class names (modify if needed)
    # class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
    class_names = ["Step 7", "Step 1", "Step  2", "Step  3", "Step  4", "Step  5", "Step  6"]

    # Hysteresis buffer
    last_predictions = deque(maxlen=args.len_buffer)
    last_confidences = deque(maxlen=args.len_buffer)

    # Open video capture
    cap = cv2.VideoCapture(args.camera_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_data = preprocess_frame(frame, img_width, img_height, normalization=False)
        print(input_data.shape)
        min_value = np.min(input_data)
        max_value = np.max(input_data)
        print(min_value, max_value)
        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get predicted class and confidence
        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data)

        # Update hysteresis buffer
        last_predictions.append(predicted_class)
        last_confidences.append(confidence)

        # Compute mode and average confidence
        mode_result = stats.mode(last_predictions)

        mode_prediction = mode_result.mode if mode_result.mode.size > 0 else None

        avg_confidence = np.mean(last_confidences)
        predicted_label = class_names[mode_prediction] if mode_prediction is not None else "Unknown"

        # Overlay text on the original frame
        text = f"{predicted_label} ({avg_confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with overlay
        cv2.imshow("Real-time Inference", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Real-time inference using TFLite model.")
    parser.add_argument('--camera_id', type=int, default=0, help="Camera ID for video capture.")
    parser.add_argument('--model_name', type=str, default='kaggle-single-frame-final-model',
                        help="TFLite model name without extension.")
    parser.add_argument('--len_buffer', type=int, default=5, help="Hysteresis buffer length.")
    main_args = parser.parse_args()
    main(main_args)
