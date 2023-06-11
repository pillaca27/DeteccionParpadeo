import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import (
    Dropout,
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
    BatchNormalization,
)
from keras.models import load_model


def generator(
    dir,
    gen=image.ImageDataGenerator(rescale=1.0 / 255),
    shuffle=True,
    batch_size=1,
    target_size=(24, 24),
    class_mode="categorical",
):  # Target size => tamaño de las imagenes del data set
    # class_mode => Serán etiquetas 2D codificadas con un solo disparo,
    # batch_size => Tamaño de los lotes de datos
    # shuffle => Si se barajan los datos (por defecto: True) Si se establece en False,
    # ordena los datos en orden alfanumérico.
    # dir => Localizacion del data set
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode="grayscale",
        class_mode=class_mode,
        target_size=target_size,
    )


BATCH_SIZE = 32
TARGET_SIZE = (24, 24)
train_batch = generator(
    "data/train", shuffle=True, batch_size=BATCH_SIZE, target_size=TARGET_SIZE
)
valid_batch = generator(
    "data/test", shuffle=True, batch_size=BATCH_SIZE, target_size=TARGET_SIZE
)
# print("hoalaa", train_batch)
SPE = len(train_batch.classes) // BATCH_SIZE
VS = len(valid_batch.classes) // BATCH_SIZE
print(SPE, VS)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(24, 24, 1)),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 1)),
        # 32 convolution filters used each of size 3x3
        # again
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(1, 1)),
        # 64 convolution filters used each of size 3x3
        # choose the best features via pooling
        # randomly turn neurons on and off to improve convergence
        Dropout(0.25),
        # flatten since too many dimensions, we only want a classification output
        Flatten(),
        # fully connected to get all relevant data
        Dense(128, activation="relu"),
        # one more dropout for convergence' sake :)
        Dropout(0.5),
        # output a softmax to squash the matrix into output probabilities
        Dense(4, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=150,
    steps_per_epoch=SPE,
    validation_steps=VS,
)

model.save("models/cnnCat8.h5", overwrite=True)


# Creacion de las graficas

plt.figure(0)
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("grafic_accuracy.png")
plt.show()

plt.figure(0)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("grafic_loss.png")
plt.show()


# Confution Matrix
# 2. División de datos en conjunto de evaluación y conjunto de entrenamiento
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


array = confusion_matrix(train_batch, valid_batch)
print(array)
