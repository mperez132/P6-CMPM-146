from operator import mod
from keras.backend import relu, sigmoid, variable
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define a sequential model here
model = models.Sequential()
# add more layers to the model...

# --- Ian's Pseudocode ---
# 11/04/21 
# I was trying to figure out how to add layers to the
# model, but wasn't sure which layers to add specifically
# and how we are reducing the grid at each layer

# single flatten layer
# "You will be flattening that into a single 1D vector"

# input_layer = layers.Input(shape=(150,))

# first_convolutional_layer = layers.C (filters=1, kernel_size=(7, 7), input_shape=(150, 150, 3))
# relu(first_convolutional_layer)
# model.add(first_convolutional_layer)
# single_flatten_layer = layers.Flatten(first_convolutional_layer)
# model.add(single_flatten_layer)

# # "... which you can run into a densly connected layer which is that fully connected
# # layer we talked about"
# maxpool_layer_1 = layers.MaxPooling2D(pool_size=(75, 75), strides=3, padding='valid')
# maxpool_layer_2 = layers.MaxPooling2D(pool_size=())

# # "from there you will get a single number that will essentially be a prediction"
# hidden_densely_connected_layer = layers.Dense(1)
# densely_connected_layer = layers.Dense(2)

# model.add(single_flatten_layer)
# model.add(hidden_densely_connected_layer)
# model.add(densely_connected_layer)

# relu(single_flatten_layer)
# relu(hidden_densely_connected_layer)
# sigmoid(densely_connected_layer)

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Finally, train this compiled model by running:
# python train.py