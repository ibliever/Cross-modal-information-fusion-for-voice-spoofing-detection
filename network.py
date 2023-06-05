import tensorflow as tf
from tensorflow.keras import layers
def LeNet():

    ip = tf.keras.layers.Input((64,64,1))
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same')(ip)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    flatten = tf.keras.layers.Flatten()(x)

    bn = tf.keras.layers.BatchNormalization()(flatten)


    dense = tf.keras.layers.Dense(120)(bn)
    dense = tf.keras.layers.LeakyReLU()(dense)
    dense = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(dense)

    model = tf.keras.Model(ip, dense)


    return model

def LeNetForAudio():

    ip = tf.keras.layers.Input((598, 257, 2))
    x = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same')(ip)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    flatten = tf.keras.layers.Flatten()(x)

    bn = tf.keras.layers.BatchNormalization()(flatten)

    dense = tf.keras.layers.Dense(120)(bn)
    dense = tf.keras.layers.LeakyReLU()(dense)
    dense = tf.keras.layers.Dense(2, activation='softmax')(dense)

    model = tf.keras.Model(ip, dense)


    return model


def VGG():
    ip = tf.keras.layers.Input((64, 64, 1))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(ip)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(x)

    flatten = tf.keras.layers.Flatten()(x)

    dense = tf.keras.layers.Dense(256, activation='relu')(flatten)
    dense = tf.keras.layers.Dense(128, activation='relu')(dense)
    dense = tf.keras.layers.Dense(2, activation='softmax')(dense)

    model = tf.keras.Model(ip, dense)
    return model

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output


class prepare(layers.Layer):
    def __init__(self):
        super(prepare, self).__init__()
        self.conv1 = layers.Conv2D(64, (3, 3), strides=1, padding="same")
        self.bn = layers.BatchNormalization()
        self.Relu = layers.Activation('relu')
        self.mp = layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.Relu(x)
        x = self.mp(x)
        return x

def ResNet18(num_classes):
    input_image = layers.Input(shape=(64, 64, 1), dtype="float32")
    output = prepare()(input_image)
    output = BasicBlock(64)(output)
    output = layers.Dropout(0.5)(output)
    output = BasicBlock(64)(output)
    output = BasicBlock(128, 2)(output)
    output = layers.Dropout(0.5)(output)
    output = BasicBlock(128)(output)
    output = BasicBlock(256, 2)(output)
    output = BasicBlock(256)(output)
    output = BasicBlock(512, 2)(output)
    output = BasicBlock(512)(output)
    output = layers.GlobalAveragePooling2D()(output)
    output = layers.Dense(num_classes)(output)
    output = layers.Activation(tf.nn.log_softmax)(output)
    return tf.keras.Model(inputs=input_image, outputs=output)

def Softmax():
    ip = tf.keras.layers.Input((4096,))
    out = tf.keras.layers.Dense(2, activation='softmax')(ip)
    #out = tf.nn.log_softmax(logits=out,axis=1)
    model = tf.keras.Model(ip, out)
    return  model


def build_model(audio_shape):
    ip = tf.keras.layers.Input(shape=audio_shape)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(ip)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=(2, 1))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=(2, 1))(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=(2, 1))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=(2, 1))(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=1, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=2, padding="VALID", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=2, padding="VALID")(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(6, 1), strides=1, padding="VALID")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    flatten = tf.keras.layers.Flatten()(x)

    dense = tf.keras.layers.Dense(4096, activation="relu")(flatten)
    dense = tf.keras.layers.Dense(4096)(dense)

    model = tf.keras.Model(ip, dense)
    return model