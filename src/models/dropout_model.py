from operator import mod
from keras.backend import relu, sigmoid, variable
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers.core import Dropout


# Define your dropout model here
model = models.Sequential()

model.add(layers.Convolution2D(32, kernel_size=(7, 7), activation=relu, padding='same', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 3))
model.add(layers.Convolution2D(64, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Convolution2D(128, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(layers.MaxPooling2D(2, 1))
model.add(layers.Convolution2D(256, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(layers.MaxPooling2D(3, 1))
model.add(layers.Convolution2D(512, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(layers.MaxPooling2D(3, 3))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation=relu))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation=sigmoid))

print(len(model.weights))
print(model.layers)
print(model.summary())

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Train this compiled model by modifying basic_train 
# to import this model, then run:
#   python train.py