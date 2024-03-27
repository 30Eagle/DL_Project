import os
from flask import Flask, render_template, request
import Retionpathypreprocess_Data as dp
import rnn_classification as rm
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model
input_shape = (128, 128, 3)
num_classes = 7
R_Model = rm.DeepANN()
model = R_Model.rnn_model(input_shape)
model.load_weights('RNNModel.keras')

@app.route('/')
def index():
    return render_template('model_selector.html')

@app.route('/run-model', methods=['POST'])
def run_model():
    images_folder_path = 'D:\\DL Project\\emotion'

    imdata = dp.PreProcess_Data()
    retina_df, train, label = imdata.preprocess(images_folder_path)
    imdata.visualization_images(images_folder_path, 7)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(train, label)

    selected_model = request.form['model']

    if selected_model == 'model1':
        # Evaluate the pre-trained model
        test_loss, test_acc = model.evaluate(tr_gen)

        # Prepare results for display
        result = f'Test Accuracy: {test_acc}'
        # You can include other result metrics here

        return render_template('model_results.html', result=result)
    elif selected_model == 'model2':
        # You can add more models here
        pass
    else:
        return "Invalid model selected"

if __name__ == "__main__":
    app.run(debug=True)
