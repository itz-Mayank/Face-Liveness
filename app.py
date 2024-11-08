from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained face liveness model
model = load_model('model.h5')

# Initialize the webcam for live video feed
video_capture = cv2.VideoCapture(0)

def get_predictions(frame):
    """
    Process a single frame, resize it, convert it to grayscale, reshape it, 
    and make a prediction using the pre-trained model.
    """
    # Resize the frame to the model's input size (90x90 in this case)
    resized_frame = cv2.resize(frame, (90, 90))
    
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Expand dimensions to match the model input shape (batch size, height, width, channels)
    reshaped_frame = np.expand_dims(grayscale_frame, axis=-1)  # Add a channel dimension (1 for grayscale)
    
    # Further expand the dimensions to add batch size (1 for a single frame)
    reshaped_frame = np.expand_dims(reshaped_frame, axis=0)
    
    # Normalize the frame if necessary (if the model was trained with normalized data)
    reshaped_frame = reshaped_frame / 255.0
    
    # Make prediction using the model
    prediction = model.predict(reshaped_frame)
    
    return prediction

def generate_frames():
    """
    Capture frames from the webcam, make predictions, and display them
    on the webpage.
    """
    while True:
        # Read a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Get prediction (e.g., liveness score)
        prediction = get_predictions(frame)

        # Display the prediction on the frame
        label = f"Prediction: {'Liveness' if prediction > 0.5 else 'Spoof'}"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format for displaying in the browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to be rendered in the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    """
    Render the main HTML page where the live video feed will be displayed.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Serve the live video feed with face liveness predictions.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the image prediction for face liveness through the '/predict' route.
    """
    data = request.get_json()
    image_data = data['image']
    
    # Convert image data from base64 (if necessary)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    prediction = get_predictions(image)

    # Return the result
    return jsonify({'prediction': prediction[0].tolist()})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
