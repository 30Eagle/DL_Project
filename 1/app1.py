import base64

from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image

app = Flask(__name__)

# pre-trained model
loaded_model = tf.keras.models.load_model('gender_detection_model.h5')

def preprocess_image(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_gender(img_array):
    prediction = loaded_model.predict(img_array)
    gender = "Male" if prediction[0][0] > 0.5 else "Female"
    return gender

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Display the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index_camer.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_capture', methods=['POST'])
def submit_capture():
    captured_image_data = request.form['capturedImage'].split(',')[1]  # Extract image data from base64 string
    captured_image_data = bytes(captured_image_data, 'utf-8')
    img_array = preprocess_image(cv2.imdecode(np.frombuffer(base64.b64decode(captured_image_data), np.uint8), -1))

    # Perform gender prediction
    gender = predict_gender(img_array)

    return jsonify({'gender': gender})

if __name__ == '__main__':
    app.run(debug=True)
