import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import math

class PreProcess_Data:
    def visualization_images(self, dir_path, nimages_per_class):
        num_classes = len(os.listdir(dir_path))
        fig, axs = plt.subplots(num_classes, nimages_per_class, figsize=(12, 12))
        dpath = dir_path
        for i, class_name in enumerate(os.listdir(dpath)):
            train_class = os.listdir(os.path.join(dpath, class_name))
            for j in range(nimages_per_class):
                img_path = os.path.join(dpath, class_name, train_class[j])
                img = cv2.imread(img_path)
                axs[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[i, j].set_title(class_name)
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.show()


    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images : {}\n'.format(len(train)))
        print('Number of train images labels: {}\n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        print(retina_df)
        return retina_df, train, label

    def generate_train_test_images(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        print(retina_df)
        train_data, test_data = train_test_split(retina_df, test_size=0.2)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.15
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128), #32 for vgg else 28 normally, 128 for rnn
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='training'
        )

        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='validation'
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32
        )

        print(f"Train images shape: {train_data.shape}")
        print(f"Testing images shape: {test_data.shape}")

        return train_generator, test_generator, validation_generator