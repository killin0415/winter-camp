import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Model
from PIL import Image
import os
from data_process import min_size, vis_training

DATAPATH = "D:\\dataset\\covid-data\\"
path = DATAPATH + "train"
path_test = DATAPATH + "test"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
BATCH_SIZE = 16
FREEZE_LAYERS = 3
NUM_EPOCHS = 20
    
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

tf.keras.backend.clear_session()
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

x = base_model.output

x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)



# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
model = Model(inputs=base_model.input, outputs=output_layer)
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,
                  optimizer=optimiser, metrics=["accuracy"])

train_log = model.fit(train_ds,
                      validation_data=val_ds,
                      batch_size=BATCH_SIZE,
                      epochs=20)

vis_training(train_log, start=1)

optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss,
                  optimizer=optimiser, metrics=["accuracy"])

train_log = model.fit(train_ds,
                      validation_data=val_ds,
                      batch_size=BATCH_SIZE,
                      epochs=10)

vis_training(train_log, start=1)

image_path = "../input/chest-xray-covid19-pneumonia/Data/test/COVID19/COVID19(471).jpg"
image_path2 = "../input/chest-xray-covid19-pneumonia/Data/test/NORMAL/NORMAL(1281).jpg"
image_path3 = "../input/chest-xray-covid19-pneumonia/Data/test/PNEUMONIA/PNEUMONIA(3433).jpg"
merged_path = [image_path,image_path2,image_path3]


for img_path in merged_path:
    img = keras.preprocessing.image.load_img(
        img_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
model.save("covid-model")