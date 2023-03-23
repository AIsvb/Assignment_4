# Loading libraries
from VisualisationFunctions import visualise_prediction

from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Softmax, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import fashion_mnist

# Preparing the data
dataset = fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing the images
train_images = train_images / 255
test_images = test_images / 255

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Building the CNN
layer_1 = Conv2D(input_shape=(28, 28, 1), filters = 10, kernel_size=(5, 5), strides=(1, 1), activation="relu")
layer_2 = AveragePooling2D(pool_size=(2, 2))
layer_3 = Conv2D(input_shape=(28, 28 ,1), filters = 25, kernel_size=(3, 3), strides=(1, 1), activation="relu")
layer_4 = BatchNormalization()
layer_5 = MaxPooling2D(strides=(2, 2))
layer_6 = Flatten()
layer_7 = Dense(70, activation="relu")
layer_8 = Dense(10, activation="softmax")

cnn = Sequential([layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8])
cnn.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Training the model
cnn.fit(train_images, train_labels, batch_size=75, epochs=10)

# Validating the model
test_loss, test_acc = cnn.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Predicting the label of the test images
predictions = cnn.predict(test_images)

# Visualising individual images and there predicted label
index = 212
visualise_prediction(test_images[index], predictions[index], test_labels[index], class_names)