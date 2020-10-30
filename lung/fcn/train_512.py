import numpy as np
import matplotlib.pyplot as plt
import time, sys

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as keras

# define dice coefficient method
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

today = time.strftime('%y%m%d', time.localtime(time.time()))

# load dataset
x_train = np.load('../dataset/CXR/data/x_train.npy')
y_train = np.load('../dataset/CXR/data/y_train.npy')
x_val = np.load('../dataset/CXR/data/x_val.npy')
y_val = np.load('../dataset/CXR/data/y_val.npy')
x_test = np.load('../dataset/CXR/data/x_test.npy')
y_test = np.load('../dataset/CXR/data/y_test.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


# build FCN model
inputs = Input(shape=(512, 512, 1))

net = Conv2D(16, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(256, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(512, activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(256, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(32, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)

# model compile
model.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss], metrics=[dice_coef])

model.summary()

# model fit
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=8, verbose=2, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, mode='auto', min_lr=1e-05)
])

# plot accuracy graph
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('dice_coef')
ax[0, 1].plot(history.history['dice_coef'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_dice_coef')
ax[1, 1].plot(history.history['val_dice_coef'], 'b--')

plt.savefig('test_image/' + today + '/test512.png')

# evaluate model with test dataset
results = model.evaluate(x_test, y_test, batch_size=8)

print(' - test_' + model.metrics_names[0] + ': ' + str(results[0]) + ' - test_' + model.metrics_names[1] + ': ' + str(results[1]))

# plot some predict results
preds = model.predict(x_test[0:10])

fig, ax = plt.subplots(len(x_test[0:10]), 3, figsize=(10, 100))

for i, pred in enumerate(preds):
    ax[i, 0].imshow(x_test[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')

plt.savefig('test_image/' + today + '/show512.png')

# save model
model_json = model.to_json()
with open(today + "_model.json", "w") as json_file :
        json_file.write(model_json)

model.save_weights(today + "_model.h5")
