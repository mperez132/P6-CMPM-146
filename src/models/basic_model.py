from operator import mod
from keras.backend import relu, sigmoid, variable
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.util.nest import _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# single flatten layer
# "You will be flattening that into a single 1D vector"
# 150 = 5 . 5 . 2 . 3
model.add(layers.Convolution2D(32, kernel_size=(7, 7), activation=relu, padding='same', input_shape=(150, 150, 3)))
# "hidden" densly connected layer
# "... which you can run into a densly connected layer which is that fully connected
# layer we talked about"
model.add(MaxPooling2D(2, 3))
model.add(layers.Convolution2D(64, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(layers.Convolution2D(128, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(MaxPooling2D(2, 1))
model.add(layers.Convolution2D(256, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(MaxPooling2D(3, 1))
model.add(layers.Convolution2D(512, kernel_size=(7, 7), activation=relu, padding='same'))
model.add(MaxPooling2D(3, 3))
# final densly connected layer
# "from there you will get a single number that will essentially be a prediction"
model.add(layers.Flatten())
model.add(layers.Dense(512, activation=relu))
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

# Finally, train this compiled model by running:
# python train.py