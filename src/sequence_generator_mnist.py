"""
https://github.com/iamhamzaabdullah/machine_learning/blob/master/Keras/data_generator.py
https://github.com/keras-team/keras/issues/9707
"""
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(len(X_train), len(X_test))
    labels_train = {index: value for index, value in enumerate(y_train)}
    labels_test = {index: value for index, value in enumerate(y_test)}
    return X_train, X_test, labels_train, labels_test


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, sequences, list_IDs, labels, batch_size=16,
                 dim=(28, 28), n_channels=1,
                 n_classes=10, shuffle=True):

        self.sequences = sequences.reshape(sequences.shape[0], 28, 28, 1)
        self.sequences = self.sequences.astype('float32')
        self.sequences /= 255

        self.dim = dim
        self.img_w, self.img_h = self.dim
        self.batch_size = batch_size
        self.labels = labels

        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        X = np.empty(
            (self.batch_size, self.img_w, self.img_h, self.n_channels),
            dtype=np.float32
        )
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i, ] = self.sequences[ID]
            Y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(Y, num_classes=self.n_classes)


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=[28, 28, 1]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    X_train, X_test, labels_train, labels_test = get_data()
    train_set = DataGenerator(X_train, range(60000), labels_train)
    test_set = DataGenerator(X_test, range(10000), labels_test)

    model = baseline_model()
    model.compile(
        loss='categorical_crossentropy', optimizer='adadelta',
        metrics=['accuracy']
    )

    model.fit_generator(
        train_set, steps_per_epoch=60000 // 16, nb_epoch=2,
        validation_data=test_set,
        validation_steps=10000 // 16
    )
