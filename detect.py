import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import io
import requests
import json
from PIL import Image
from tqdm import tqdm

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


df = pd.read_json("Indian_Number_plates.json", lines=True)
df.head()

dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

counter = 0
for index, row in tqdm(df.iterrows()):

    dataset["image_name"].append("licensed_car{}".format(counter))

    data = row["annotation"]

    dataset["image_width"].append(data[0]["imageWidth"])
    dataset["image_height"].append(data[0]["imageHeight"])
    dataset["top_x"].append(data[0]["points"][0]["x"])
    dataset["top_y"].append(data[0]["points"][0]["y"])
    dataset["bottom_x"].append(data[0]["points"][1]["x"])
    dataset["bottom_y"].append(data[0]["points"][1]["y"])

    counter += 1
print("Downloaded {} car images.".format(counter))

# creating a datafram object from the dictionary
df = pd.DataFrame(dataset)
df.head()

df = pd.read_csv("indian_license_plates.csv")
df["image_name"] = df["image_name"] + ".jpeg"
df.drop(["image_width", "image_height"], axis=1, inplace=True)
df.head()

test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(test_samples, axis=0)
WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def show_img(index):
    image = cv2.imread("Indian Number Plates/Indian Number Plates/" + df["image_name"].iloc[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    tx = int(df["top_x"].iloc[index] * WIDTH)
    ty = int(df["top_y"].iloc[index] * HEIGHT)
    bx = int(df["bottom_x"].iloc[index] * WIDTH)
    by = int(df["bottom_y"].iloc[index] * HEIGHT)

    #image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()

show_img(144)
show_img(17)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory="Indian Number Plates/Indian Number Plates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32,
    class_mode="other",
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory="Indian Number Plates/Indian Number Plates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32,
    class_mode="other",
    subset="validation")

model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()

STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))
STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))

print("Train step size:", STEP_SIZE_TRAIN)
print("Validation step size:", STEP_SIZE_VAL)

train_generator.reset()
validation_generator.reset()

adam = Adam(learning_rate=0.0005)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

#The Adam optimiser is used here to find our losses.
history = model.fit(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=10)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();
model.evaluate(validation_generator, steps=STEP_SIZE_VAL)

#the evaluate() function will return the list of metrics that the model was compiled with.

predictions = model.predict(validation_generator)
print(predictions)
for idx, row in df.iloc[test_samples].iterrows():
    img = cv2.resize(cv2.imread("Indian Number Plates/Indian Number Plates/" + row[0]) / 255.0, dsize=(WIDTH, HEIGHT))
    y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH

    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]

    start_point = (int(np.float32(xt)),int(np.float32(yt)))
    print(start_point)
    end_point = (int(np.float32(xb)),int(np.float32(yb)))
    print(end_point)
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()

WIDTH = 224
HEIGHT = 224
CHANNEL = 3


def show_number(index):
    image = cv2.imread("Indian Number Plates/Indian Number Plates/" + df["image_name"].iloc[index])
    img = cv2.resize(image / 255.0, dsize=(WIDTH, HEIGHT))
    # image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # im_cr = img[int(yt):int(yb),int(xt):int(xb)]

    url_api = "https://api.ocr.space/parse/image"
    _, compressedimag = cv2.imencode(".jpeg", image, [1, 90])  # enccoding image
    file_byte = io.BytesIO(compressedimag)
    result = requests.post(url_api,
                           files={"image.jpeg": file_byte},
                           data={"apikey": "c56514c74b88957"})

    result = (result.content.decode())  # decoding the result
    result = json.loads(result)
    final = result.get("ParsedResults")[0].get("ParsedText")  # traversing through the parsed text
    # final1 = result.get("ParsedResults")[0]
    # print(result)
    # print(final)
    # print(final1)
    plt.imshow(image)
    plt.show()
    if (final == ""):
        print("UNABLE TO READ.")
    else:
        print(final)
        num_plate = final
        num_plate = num_plate.strip()
        print("VEHICLE NUMBER: "+num_plate+"")

        if num_plate == 'HR 26 CH 3604':#146
            print("The vehicle belongs to Mr.AANANDH.\n SLOT : ALLOWED")

        elif num_plate == 'MH02CT2727':#123 #'MH20EE 7598':#54
            print("The vehicle belongs to Mr.BABU.\n SLOT : ALLOWED")

        elif num_plate == 'DL 3 CAY 9324':#16
            print("The vehicle belongs to Mr.KAMAL.\n SLOT : ALLOWED")

        elif num_plate == 'TN66U8215':#109
            print("The vehicle belongs to Mr.KARTHIK.\n SLOT : ALLOWED")

        elif num_plate == 'vKA51MJ8156':#5
            print("The vehicle belongs to Mr.SELVAM.\n SLOT : ALLOWED")

        else:
            print("Not Allowed")


#71


def plate_show():
    index = int(input("Enter index: "))
    image = cv2.imread("Indian Number Plates/Indian Number Plates/" + df["image_name"].iloc[index])
    img = cv2.resize(image / 255.0, dsize=(WIDTH, HEIGHT))
    y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH

    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]
    start_point = (int(np.float32(xt)), int(np.float32(yt)))
    print(start_point)
    end_point = (int(np.float32(xb)), int(np.float32(yb)))
    print(end_point)
    print("The Number Plate Detected is :")



    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, start_point, end_point, (0, 0, 255), 1)
    im_cr = image[int(yt):int(yb), int(xt):int(xb)]
    plt.imshow(image)
    plt.show()
    show_number(index)
plate_show()
