import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Retionpathypreprocess_Data as dp
import rnn_classification as rm

if __name__ == "__main__":
    images_folder_path = 'D:\\DL Project\\emotion'

    imdata = dp.PreProcess_Data()
    retina_df, train, label = imdata.preprocess(images_folder_path)
    imdata.visualization_images(images_folder_path, 7)
    tr_gen, tt_gen, va_gen = imdata.generate_train_test_images(train, label)

    input_shape = (128, 128, 3)
    num_classes = 7
    R_Model = rm.DeepANN()
    m = R_Model.rnn_model(input_shape)
    Rnn_history = m.fit(tr_gen, epochs=10, validation_data=va_gen)

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(Rnn_history.history['loss'], label='Training Loss')
    plt.plot(Rnn_history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



    RNN_test_loss, RNN_test_acc = m.evaluate(tr_gen)
    print(f'Test Accuracy: {RNN_test_acc}')
    m.save('RNNModel.keras')
    print(m.summary())