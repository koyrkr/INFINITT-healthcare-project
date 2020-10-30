import numpy as np
import matplotlib.pyplot as plt
import time

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, UpSampling2D, concatenate, Conv2DTranspose
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as keras

today = time.strftime('%y%m%d', time.localtime(time.time()))

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

x_train = np.load('../dataset/CXR/data/x_train.npy')
y_train = np.load('../dataset/CXR/data/y_train.npy')
x_val = np.load('../dataset/CXR/data/x_val.npy')
y_val = np.load('../dataset/CXR/data/y_val.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

inputs = Input(shape=(512, 512, 1))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=8, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, epsilon=0.0001, cooldown=2, verbose=1, mode='auto', min_lr=1e-05)
])

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['binary_accuracy'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_binary_accuracy'], 'b--')

plt.savefig('test_image/test512.png')

preds = model.predict(x_val[0:10])

fig, ax = plt.subplots(len(x_val[0:10]), 3, figsize=(10, 100))

for i, pred in enumerate(preds):
    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')

plt.savefig('test_image/show512.png')

model_json = model.to_json()
with open(today + "_model.json", "w") as json_file :
        json_file.write(model_json)

model.save_weights(today + "_model.h5")
