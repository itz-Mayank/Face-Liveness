import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = "Real-time-Facial-Recognition-with-Object-Detection-using-TensorFlow-and-OpenCV.-main/model.h5"  # Update with your model's actual path
model = load_model(model_path)

# Define the input shape based on the model's expectation
EXPECTED_WIDTH = 3
EXPECTED_HEIGHT = 90
EXPECTED_CHANNELS = 1  # Assuming grayscale; change to 3 if the model expects RGB

# Class labels - update these to match your model's classes
class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera; change if using external webcam

if not cap.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("Press 'q' to quit the webcam stream.")

# Capture frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture an image from the webcam.")
        break
    
    # Display the original color frame
    cv2.imshow("Webcam Feed - Color", frame)

    # Prepare a grayscale version for model prediction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(gray_frame, (EXPECTED_WIDTH, EXPECTED_HEIGHT))

    # Reshape the resized grayscale frame to match model's input dimensions
    frame_resized = np.expand_dims(frame_resized, axis=-1)  # Add channel dimension
    frame_array = np.expand_dims(frame_resized, axis=0)     # Add batch dimension

    # Predict using the model
    try:
        predictions = model.predict(frame_array)[0]  # Get the prediction array
        predicted_class = np.argmax(predictions)  # Find the index of the highest probability
        confidence = predictions[predicted_class] * 100  # Convert to percentage

        # Display detailed prediction information
        print("Detailed Prediction Probabilities (in %):")
        for i, prob in enumerate(predictions):
            print(f"{class_labels[i]}: {prob * 100:.2f}%")

        # Display the predicted class and confidence score on the frame
        label_text = f"Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2f}%)"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Optional: Print the summary prediction
        print(label_text)
        
    except ValueError as e:
        print(f"Error during prediction: {e}")
        break
    
    # Show the frame with the prediction label
    cv2.imshow("Webcam Feed - Prediction", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
