from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = keras.applications.VGG16(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False)

# Freeze base model
base_model.trainable = False

# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False
)

# load and iterate training dataset
train_it = datagen.flow_from_directory('fruits/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='categorical')
                                    
# load and iterate validation dataset
valid_it = datagen.flow_from_directory('fruits/valid/', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")


model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=50)

# Unfreeze the base model
base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=20)

model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)

'''
Evaluating model 5 times to obtain average accuracy...

11/10 [================================] - 4s 345ms/step - loss: 0.0986 - accuracy: 0.9818
11/10 [================================] - 4s 344ms/step - loss: 0.1158 - accuracy: 0.9848
11/10 [================================] - 4s 345ms/step - loss: 0.0585 - accuracy: 0.9909
11/10 [================================] - 4s 346ms/step - loss: 0.0538 - accuracy: 0.9909
11/10 [================================] - 4s 347ms/step - loss: 0.0711 - accuracy: 0.9909

Accuracy required to pass the assessment is 0.92 or greater.
Your average accuracy is 0.9878.

Congratulations! You passed the assessment!
See instructions below to generate a certificate.
'''