from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

#  pre-trained model
loaded_model = tf.keras.models.load_model('gender_detection_model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('index.html', error='Invalid file extension')

        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = loaded_model.predict(img_array)

        gender = "Male" if prediction[0][0] > 0.5 else "Female"

        return render_template('result.html', image_path=file_path, gender=gender)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
