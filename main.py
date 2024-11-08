import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model (replace with your model's path)
model = tf.keras.models.load_model(r'C:\Users\Mayank Meghwal\Desktop\Face Liveness\Face-liveness\model.h5')


# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the face to match the model's input format
def preprocess_face(face):
    """
    Preprocess the face to match the input format expected by the model.
    """
    # Resize the face to the model's expected input size (90x90)
    face_resized = cv2.resize(face, (90, 90))  # Resize to 90x90
    
    # Convert to grayscale (if not already grayscale)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to 0-1
    face_gray = face_gray / 255.0
    
    # Add extra dimension to make it (90, 90, 1)
    face_gray = np.expand_dims(face_gray, axis=-1)  # Shape (90, 90, 1)
    
    # Repeat the single channel to make it (90, 90, 3)
    face_rgb = np.repeat(face_gray, 3, axis=-1)  # Shape (90, 90, 3)
    
    # Add batch dimension (convert to shape (1, 90, 90, 3))
    face_rgb = np.expand_dims(face_rgb, axis=0)
    
    return face_rgb

# Open the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the captured frame to RGB (for consistent color processing)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = img_rgb[y:y+h, x:x+w]
        
        # Preprocess the face to match model input
        face_preprocessed = preprocess_face(face)
        
        # Make prediction using the pre-trained model
        prediction = model.predict(face_preprocessed)
        
        # Display the prediction result (you can adjust this depending on your model output)
        print(f"Prediction: {prediction}")
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting frame with detections
    cv2.imshow('Real-time Face Detection', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
