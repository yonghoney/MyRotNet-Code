import tensorflow as tf
import numpy as np
from tqdm import tqdm
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img, train_labels), (val_img, val_labels) = fashion_mnist.load_data()


def rotation(Img0):
    Img90 = tf.image.rot90(Img0)
    Img180 = tf.image.rot90(Img0, k = 2)
    Img270 = tf.image.rot90(Img0, k = 3)
    return(np.asarray(Img90), np.asarray(Img180), np.asarray(Img270))

img0 = np.zeros(shape = (1, 28, 28))
img90 = np.zeros(shape = (1, 28, 28))
img180 = np.zeros(shape = (1, 28, 28))
img270 = np.zeros(shape = (1, 28, 28))

for i, j in tqdm(enumerate(train_img), total = len(train_img)):
    tmp90, tmp180, tmp270 = rotation(j.reshape([28, 28, 1]))
    img0 = np.vstack([img0, j.reshape([1, 28, 28])])
    img90 = np.vstack([img90, tmp90.reshape([1, 28, 28])])
    img180 = np.vstack([img180, tmp180.reshape([1, 28, 28])])
    img270 = np.vstack([img270, tmp270.reshape([1, 28, 28])])

train_img90 = img90[1:]
train_img180 = img180[1:]
train_img270 = img270[1:]

img0 = np.zeros(shape = (1, 28, 28))
img90 = np.zeros(shape = (1, 28, 28))
img180 = np.zeros(shape = (1, 28, 28))
img270 = np.zeros(shape = (1, 28, 28))

for i, j in tqdm(enumerate(val_img), total = len(val_img)):
    tmp90, tmp180, tmp270 = rotation(j.reshape([28, 28, 1]))
    img0 = np.vstack([img0, j.reshape([1, 28, 28])])
    img90 = np.vstack([img90, tmp90.reshape([1, 28, 28])])
    img180 = np.vstack([img180, tmp180.reshape([1, 28, 28])])
    img270 = np.vstack([img270, tmp270.reshape([1, 28, 28])])

val_img90 = img90[1:]
val_img180 = img180[1:]
val_img270 = img270[1:]

train_img = np.stack([train_img, train_img90, train_img180, train_img270])
val_img = np.stack([val_img, val_img90, val_img180, val_img270])

train_y = np.ones((len(train_img)*train_img.shape[1], 4), dtype = 'int')
val_y = np.ones((len(val_img)*val_img.shape[1], 4), dtype = 'int')

train_y[:60000] = [1, 0, 0, 0]
train_y[60000:120000] = [0, 1, 0, 0]
train_y[120000:180000] = [0, 0, 1, 0]
train_y[180000:] = [0, 0, 0, 1]

val_y[:10000] = [1, 0, 0, 0]
val_y[10000:20000] = [0, 1, 0, 0]
val_y[20000:30000] = [0, 0, 1, 0]
val_y[30000:] = [0, 0, 0, 1]

np.save('train_mnist_fashion.npy', train_img)
np.save('val_mnist_fashion.npy', val_img)


import numpy as np
train_img = np.load('train_mnist_fashion.npy')
val_img = np.load('val_mnist_fashion.npy')
train_y = np.ones((len(train_img)*train_img.shape[1], 4), dtype = 'int')
val_y = np.ones((len(val_img)*val_img.shape[1], 4), dtype = 'int')

train_y[:60000] = [1, 0, 0, 0]
train_y[60000:120000] = [0, 1, 0, 0]
train_y[120000:180000] = [0, 0, 1, 0]
train_y[180000:] = [0, 0, 0, 1]

val_y[:10000] = [1, 0, 0, 0]
val_y[10000:20000] = [0, 1, 0, 0]
val_y[20000:30000] = [0, 0, 1, 0]
val_y[30000:] = [0, 0, 0, 1]

train_img = train_img.reshape([240000, 28, 28, 1])
val_img = val_img.reshape([40000, 28, 28, 1])




import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img2, train_labels), (val_img2, val_labels) = fashion_mnist.load_data()
train_img2 = train_img2.reshape([60000, 28, 28, 1])
val_img2 = val_img2.reshape([10000, 28, 28, 1])


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPool2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

batch_size = 128
epochs = 30
num_classes = 10
weight_decay = 1e-6
nets = 15


def NIN(model, cnn1, cnn2, cnn3, inputshape=False):
    weight_decay = 1e-6
    if inputshape == True:
        input_shape = (28, 28, 1)
        model.add(Conv2D(cnn1, (5, 5), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal', input_shape=input_shape))
    else:
        model.add(Conv2D(cnn1, (5, 5), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(cnn2, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(cnn3, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    return model

def rotnet(n_blocks = 3):
    tf.random.set_seed(1234)
    model = Sequential()
    # block 1
    model = NIN(model, 192, 160, 96, inputshape=True)
    model.add(Dropout(0.2))
    # block 2~n-1
    for n in range(n_blocks - 2):
        model = NIN(model, 192, 192, 192)
        model.add(Dropout(0.2))
    # block n
    model = NIN(model, 192, 192, 192)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(4, activation='linear'))
    model.add(Activation('softmax'))
    # opt = SGD(lr=0.1, decay = 5e-4, momentum = 0.9)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

# freeze block 'n_blocks'
def freeze_net(model, block, n_class, trainable = False, classifier = 'linear'):
    tf.random.set_seed(1234)
    freeze_model = Sequential()
    for i in range(11 * block - 2):
        model.layers[i].trainable = trainable
        freeze_model.add(model.layers[i])
    if classifier == 'linear':
        freeze_model.add(GlobalAveragePooling2D())
        freeze_model.add(Dense(200, activation='relu'))
        freeze_model.add(BatchNormalization())
        freeze_model.add(Dense(200, activation='relu'))
        freeze_model.add(BatchNormalization())
        freeze_model.add(Dense(n_class, activation='softmax'))
    else:
        freeze_model.add(Dropout(0.2))
        freeze_model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
        freeze_model.add(BatchNormalization())
        freeze_model.add(Activation('relu'))
        freeze_model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
        freeze_model.add(BatchNormalization())
        freeze_model.add(Activation('relu'))
        freeze_model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal'))
        freeze_model.add(BatchNormalization())
        freeze_model.add(Activation('relu'))
        freeze_model.add(GlobalAveragePooling2D())
        freeze_model.add(Dense(10, activation='softmax'))
    freeze_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return freeze_model

n_blocks = 3
rotation_model3 = rotnet(n_blocks=n_blocks)
rotation_model3.fit(train_img, train_y, validation_data=(val_img, val_y), batch_size=128, epochs=30)

freeze_model3_1 = freeze_net(rotation_model3, block = 1, n_class = 10)
freeze_model3_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model3_2 = freeze_net(rotation_model3, block = 2, n_class = 10)
freeze_model3_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model3_3 = freeze_net(rotation_model3, block = 3, n_class = 10)
freeze_model3_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)


n_blocks = 4
rotation_model4 = rotnet(n_blocks=n_blocks)
rotation_model4.fit(train_img, train_y, validation_data=(val_img, val_y), batch_size=128, epochs=30)

freeze_model4_1 = freeze_net(rotation_model4, block = 1, n_class = 10)
freeze_model4_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model4_2 = freeze_net(rotation_model4, block = 2, n_class = 10)
freeze_model4_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model4_3 = freeze_net(rotation_model4, block = 3, n_class = 10)
freeze_model4_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model4_4 = freeze_net(rotation_model4, block = 4, n_class = 10)
freeze_model4_4.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)


n_blocks = 5
rotation_model5 = rotnet(n_blocks=n_blocks)
rotation_model5.fit(train_img, train_y, validation_data=(val_img, val_y), batch_size=128, epochs=30)

freeze_model5_1 = freeze_net(rotation_model5, block = 1, n_class = 10)
freeze_model5_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model5_2 = freeze_net(rotation_model5, block = 2, n_class = 10)
freeze_model5_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model5_3 = freeze_net(rotation_model5, block = 3, n_class = 10)
freeze_model5_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model5_4 = freeze_net(rotation_model5, block = 4, n_class = 10)
freeze_model5_4.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)

freeze_model5_5 = freeze_net(rotation_model5, block = 5, n_class = 10)
freeze_model5_5.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 64, epochs = 200, shuffle = True)


non_freeze3_1 = freeze_net(rotation_model3, block = 1, n_class = 10, classifier = 'non')
non_freeze3_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze3_2 = freeze_net(rotation_model3, block = 2, n_class = 10, classifier = 'non')
non_freeze3_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze3_3 = freeze_net(rotation_model3, block = 3, n_class = 10, classifier = 'non')
non_freeze3_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze4_1 = freeze_net(rotation_model4, block = 1, n_class = 10, classifier = 'non')
non_freeze4_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze4_2 = freeze_net(rotation_model4, block = 2, n_class = 10, classifier = 'non')
non_freeze4_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze4_3 = freeze_net(rotation_model4, block = 3, n_class = 10, classifier = 'non')
non_freeze4_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze4_4 = freeze_net(rotation_model4, block = 4, n_class = 10, classifier = 'non')
non_freeze4_4.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze5_1 = freeze_net(rotation_model5, block = 1, n_class = 10, classifier = 'non')
non_freeze5_1.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze5_2 = freeze_net(rotation_model5, block = 2, n_class = 10, classifier = 'non')
non_freeze5_2.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze5_3 = freeze_net(rotation_model5, block = 3, n_class = 10, classifier = 'non')
non_freeze5_3.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze5_4 = freeze_net(rotation_model5, block = 4, n_class = 10, classifier = 'non')
non_freeze5_4.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)

non_freeze5_5 = freeze_net(rotation_model5, block = 5, n_class = 10, classifier = 'non')
non_freeze5_5.fit(val_img2[:500], val_labels[:500], validation_data = (val_img2[500:], val_labels[500:]), batch_size = 16, epochs = 200, shuffle = True)


import matplotlib.pyplot as plt
plt.plot(freeze_model3_1.history.history['val_accuracy'], label = 'model3_1')
plt.plot(freeze_model3_2.history.history['val_accuracy'], label = 'model3_2')
plt.plot(freeze_model3_3.history.history['val_accuracy'], label = 'model3_3')
plt.legend()
plt.title('comparison freeze_model3')
plt.show()

plt.plot(freeze_model4_1.history.history['val_accuracy'], label = 'model4_1')
plt.plot(freeze_model4_2.history.history['val_accuracy'], label = 'model4_2')
plt.plot(freeze_model4_3.history.history['val_accuracy'], label = 'model4_3')
plt.plot(freeze_model4_4.history.history['val_accuracy'], label = 'model4_4')
plt.legend()
plt.title('comparison freeze_model4')
plt.show()

plt.plot(freeze_model5_1.history.history['val_accuracy'], label = 'model5_1')
plt.plot(freeze_model5_2.history.history['val_accuracy'], label = 'model5_2')
plt.plot(freeze_model5_3.history.history['val_accuracy'], label = 'model5_3')
plt.plot(freeze_model5_4.history.history['val_accuracy'], label = 'model5_4')
plt.plot(freeze_model5_5.history.history['val_accuracy'], label = 'model5_5')
plt.legend()
plt.title('comparison freeze_model5')
plt.show()

plt.plot(non_freeze3_1.history.history['val_accuracy'], label = 'model3_1')
plt.plot(non_freeze3_2.history.history['val_accuracy'], label = 'model3_2')
plt.plot(non_freeze3_3.history.history['val_accuracy'], label = 'model3_3')
plt.legend()
plt.title('comparison nonlinear_model3')
plt.show()

plt.plot(non_freeze4_1.history.history['val_accuracy'], label = 'model4_1')
plt.plot(non_freeze4_2.history.history['val_accuracy'], label = 'model4_2')
plt.plot(non_freeze4_3.history.history['val_accuracy'], label = 'model4_3')
plt.plot(non_freeze4_4.history.history['val_accuracy'], label = 'model4_4')
plt.legend()
plt.title('comparison nonlinear_model4')
plt.show()

plt.plot(non_freeze5_1.history.history['val_accuracy'], label = 'model5_1')
plt.plot(non_freeze5_2.history.history['val_accuracy'], label = 'model5_2')
plt.plot(non_freeze5_3.history.history['val_accuracy'], label = 'model5_3')
plt.plot(non_freeze5_4.history.history['val_accuracy'], label = 'model5_4')
plt.plot(non_freeze5_5.history.history['val_accuracy'], label = 'model5_5')
plt.legend()
plt.title('comparison nonlinear_model5')
plt.show()


plt.plot(freeze_model3_1.history.history['val_accuracy'], label = 'linear3_1')
plt.plot(non_freeze3_1.history.history['val_accuracy'], label = 'non3_1')
plt.legend()
plt.title('comparison 3_1')
plt.show()

plt.plot(freeze_model3_2.history.history['val_accuracy'], label = 'linear3_2')
plt.plot(non_freeze3_2.history.history['val_accuracy'], label = 'non3_2')
plt.legend()
plt.title('comparison 3_2')
plt.show()

plt.plot(freeze_model3_3.history.history['val_accuracy'], label = 'linear3_3')
plt.plot(non_freeze3_3.history.history['val_accuracy'], label = 'non3_3')
plt.legend()
plt.title('comparison 3_3')
plt.show()

plt.plot(freeze_model4_1.history.history['val_accuracy'], label = 'linear4_1')
plt.plot(non_freeze4_1.history.history['val_accuracy'], label = 'non4_1')
plt.legend()
plt.title('comparison 4_1')
plt.show()

plt.plot(freeze_model4_2.history.history['val_accuracy'], label = 'linear4_2')
plt.plot(non_freeze4_2.history.history['val_accuracy'], label = 'non4_2')
plt.legend()
plt.title('comparison 4_2')
plt.show()

plt.plot(freeze_model4_3.history.history['val_accuracy'], label = 'linear4_3')
plt.plot(non_freeze4_3.history.history['val_accuracy'], label = 'non4_3')
plt.legend()
plt.title('comparison 4_3')
plt.show()

plt.plot(freeze_model4_4.history.history['val_accuracy'], label = 'linear4_4')
plt.plot(non_freeze4_4.history.history['val_accuracy'], label = 'non4_4')
plt.legend()
plt.title('comparison 4_4')
plt.show()

plt.plot(freeze_model5_1.history.history['val_accuracy'], label = 'linear5_1')
plt.plot(non_freeze5_1.history.history['val_accuracy'], label = 'non5_1')
plt.legend()
plt.title('comparison 5_1')
plt.show()

plt.plot(freeze_model5_2.history.history['val_accuracy'], label = 'linear5_2')
plt.plot(non_freeze5_2.history.history['val_accuracy'], label = 'non5_2')
plt.legend()
plt.title('comparison 5_2')
plt.show()

plt.plot(freeze_model5_3.history.history['val_accuracy'], label = 'linear5_3')
plt.plot(non_freeze5_3.history.history['val_accuracy'], label = 'non5_3')
plt.legend()
plt.title('comparison 5_3')
plt.show()

plt.plot(freeze_model5_4.history.history['val_accuracy'], label = 'linear5_4')
plt.plot(non_freeze5_4.history.history['val_accuracy'], label = 'non5_4')
plt.legend()
plt.title('comparison 5_4')
plt.show()

plt.plot(freeze_model5_5.history.history['val_accuracy'], label = 'linear5_5')
plt.plot(non_freeze5_5.history.history['val_accuracy'], label = 'non5_5')
plt.legend()
plt.title('comparison 5_5')
plt.show()

loss3_1 = freeze_model3_1.history.history['val_loss'].copy()
loss3_2 = freeze_model3_2.history.history['val_loss'].copy()
loss3_3 = freeze_model3_3.history.history['val_loss'].copy()

loss4_1 = freeze_model4_1.history.history['val_loss'].copy()
loss4_2 = freeze_model4_2.history.history['val_loss'].copy()
loss4_3 = freeze_model4_3.history.history['val_loss'].copy()
loss4_4 = freeze_model4_4.history.history['val_loss'].copy()

loss5_1 = freeze_model5_1.history.history['val_loss'].copy()
loss5_2 = freeze_model5_2.history.history['val_loss'].copy()
loss5_3 = freeze_model5_3.history.history['val_loss'].copy()
loss5_4 = freeze_model5_4.history.history['val_loss'].copy()
loss5_5 = freeze_model5_5.history.history['val_loss'].copy()

best3_1 = np.where(np.array(loss3_1) == min(loss3_1))[0][0]
best3_2 = np.where(np.array(loss3_2) == min(loss3_2))[0][0]
best3_3 = np.where(np.array(loss3_3) == min(loss3_3))[0][0]

best4_1 = np.where(np.array(loss4_1) == min(loss4_1))[0][0]
best4_2 = np.where(np.array(loss4_2) == min(loss4_2))[0][0]
best4_3 = np.where(np.array(loss4_3) == min(loss4_3))[0][0]
best4_4 = np.where(np.array(loss4_4) == min(loss4_4))[0][0]

best5_1 = np.where(np.array(loss5_1) == min(loss5_1))[0][0]
best5_2 = np.where(np.array(loss5_2) == min(loss5_2))[0][0]
best5_3 = np.where(np.array(loss5_3) == min(loss5_3))[0][0]
best5_4 = np.where(np.array(loss5_4) == min(loss5_4))[0][0]
best5_5 = np.where(np.array(loss5_5) == min(loss5_5))[0][0]

print('freeze3', freeze_model3_1.history.history['val_accuracy'][best3_1],
freeze_model3_2.history.history['val_accuracy'][best3_2],
freeze_model3_3.history.history['val_accuracy'][best3_3])

print('freeze4', freeze_model4_1.history.history['val_accuracy'][best4_1],
freeze_model4_2.history.history['val_accuracy'][best4_2],
freeze_model4_3.history.history['val_accuracy'][best4_3],
freeze_model4_4.history.history['val_accuracy'][best4_4])

print('freeze5', freeze_model5_1.history.history['val_accuracy'][best5_1],
freeze_model5_2.history.history['val_accuracy'][best5_2],
freeze_model5_3.history.history['val_accuracy'][best5_3],
freeze_model5_4.history.history['val_accuracy'][best5_4],
freeze_model5_5.history.history['val_accuracy'][best5_5])


loss3_1 = non_freeze3_1.history.history['val_loss'].copy()
loss3_2 = non_freeze3_2.history.history['val_loss'].copy()
loss3_3 = non_freeze3_3.history.history['val_loss'].copy()

loss4_1 = non_freeze4_1.history.history['val_loss'].copy()
loss4_2 = non_freeze4_2.history.history['val_loss'].copy()
loss4_3 = non_freeze4_3.history.history['val_loss'].copy()
loss4_4 = non_freeze4_4.history.history['val_loss'].copy()

loss5_1 = non_freeze5_1.history.history['val_loss'].copy()
loss5_2 = non_freeze5_2.history.history['val_loss'].copy()
loss5_3 = non_freeze5_3.history.history['val_loss'].copy()
loss5_4 = non_freeze5_4.history.history['val_loss'].copy()
loss5_5 = non_freeze5_5.history.history['val_loss'].copy()

best3_1 = np.where(np.array(loss3_1) == min(loss3_1))[0][0]
best3_2 = np.where(np.array(loss3_2) == min(loss3_2))[0][0]
best3_3 = np.where(np.array(loss3_3) == min(loss3_3))[0][0]

best4_1 = np.where(np.array(loss4_1) == min(loss4_1))[0][0]
best4_2 = np.where(np.array(loss4_2) == min(loss4_2))[0][0]
best4_3 = np.where(np.array(loss4_3) == min(loss4_3))[0][0]
best4_4 = np.where(np.array(loss4_4) == min(loss4_4))[0][0]

best5_1 = np.where(np.array(loss5_1) == min(loss5_1))[0][0]
best5_2 = np.where(np.array(loss5_2) == min(loss5_2))[0][0]
best5_3 = np.where(np.array(loss5_3) == min(loss5_3))[0][0]
best5_4 = np.where(np.array(loss5_4) == min(loss5_4))[0][0]
best5_5 = np.where(np.array(loss5_5) == min(loss5_5))[0][0]

print('freeze3', non_freeze3_1.history.history['val_accuracy'][best3_1],
non_freeze3_2.history.history['val_accuracy'][best3_2],
non_freeze3_3.history.history['val_accuracy'][best3_3])

print('freeze4', non_freeze4_1.history.history['val_accuracy'][best4_1],
non_freeze4_2.history.history['val_accuracy'][best4_2],
non_freeze4_3.history.history['val_accuracy'][best4_3],
non_freeze4_4.history.history['val_accuracy'][best4_4])

print('freeze5', non_freeze5_1.history.history['val_accuracy'][best5_1],
non_freeze5_2.history.history['val_accuracy'][best5_2],
non_freeze5_3.history.history['val_accuracy'][best5_3],
non_freeze5_4.history.history['val_accuracy'][best5_4],
non_freeze5_5.history.history['val_accuracy'][best5_5])


# rotation_model3.save('rotation3.h5')
# freeze_model3_1.save('freeze3_1.h5')
# freeze_model3_2.save('freeze3_2.h5')
# freeze_model3_3.save('freeze3_3.h5')
#
# rotation_model4.save('rotation4.h5')
# freeze_model4_1.save('freeze4_1.h5')
# freeze_model4_2.save('freeze4_2.h5')
# freeze_model4_3.save('freeze4_3.h5')
# freeze_model4_4.save('freeze4_4.h5')
#
# rotation_model5.save('rotation5.h5')
# freeze_model5_1.save('freeze5_1.h5')
# freeze_model5_2.save('freeze5_2.h5')
# freeze_model5_3.save('freeze5_3.h5')
# freeze_model5_4.save('freeze5_4.h5')
# freeze_model5_5.save('freeze5_5.h5')
#
# non_freeze3_1.save('non_freeze3_1.h5')
# non_freeze3_2.save('non_freeze3_2.h5')
# non_freeze3_3.save('non_freeze3_3.h5')
#
# non_freeze4_1.save('non_freeze4_1.h5')
# non_freeze4_2.save('non_freeze4_2.h5')
# non_freeze4_3.save('non_freeze4_3.h5')
# non_freeze4_4.save('non_freeze4_4.h5')
#
# non_freeze5_1.save('non_freeze5_1.h5')
# non_freeze5_2.save('non_freeze5_2.h5')
# non_freeze5_3.save('non_freeze5_3.h5')
# non_freeze5_4.save('non_freeze5_4.h5')
# non_freeze5_5.save('non_freeze5_5.h5')
#
# plt.imshow(train_img[0])
# plt.show()
#
# plt.imshow(train_img[60000])
# plt.show()
#
# plt.imshow(train_img[120000])
# plt.show()
#
# plt.imshow(train_img[180000])
# plt.show()