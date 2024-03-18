import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'gender//train'       # male-1944, female=1947 images
validation_dir = 'gender//valid'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

history = model.fit(train_generator, epochs=15, validation_data=validation_generator)

model.save('gender_detection_model.h5')

print("Training Accuracy:", history.history['accuracy'][-1])
print("Training Precision:", history.history['precision'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
print("Validation Precision:", history.history['val_precision'][-1])
print("Training History:",history.history)

loaded_model = tf.keras.models.load_model('gender_detection_model.h5')




 #img_paths = ['img_7.png']

# # List to store predictions
# predictions_list = []
#
# # Predict for each image
# for img_path in img_paths:
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     img_array = test_datagen.flow(img_array, shuffle=False).next()
#
#     predictions = loaded_model.predict(img_array)
#
#     if predictions[0][0] > 0.5:
#         predictions_list.append("male")
#     else:
#         predictions_list.append("female")
#
# # Print predictions for all images
# for i, img_path in enumerate(img_paths):
#     print(f"Image {i + 1}: {img_path}, Predicted Gender: {predictions_list[i]}")
