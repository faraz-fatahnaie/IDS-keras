import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, \
    BatchNormalization, Attention, Flatten, Dense
from keras.models import Model


class CNNATTENTION1D(Model):
    def __init__(self):
        super(CNNATTENTION1D, self).__init__()
        query_input = Input(shape=(196, 1), dtype='float32')

        cnn_layer = Conv1D(filters=64, kernel_size=64, strides=1, padding='same', activation='relu')(query_input)
        pool = MaxPooling1D(pool_size=4)(cnn_layer)
        norm = BatchNormalization()(pool)
        attention = Attention()([norm, norm])

        cnn_layer2 = Conv1D(filters=128, kernel_size=64, strides=1, padding='same', activation='relu')(attention)
        pool2 = MaxPooling1D(pool_size=2)(cnn_layer2)
        norm2 = BatchNormalization()(pool2)
        attention2 = Attention()([norm2, norm2])

        cnn_layer3 = Conv1D(filters=256, kernel_size=64, strides=1, padding='same', activation='relu')(attention2)
        pool3 = MaxPooling1D(pool_size=2)(cnn_layer3)
        norm3 = BatchNormalization()(pool3)
        attention3 = Attention()([norm3, norm3])

        flatten = Flatten()(attention3)
        output = Dense(2,
                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                       bias_regularizer=tf.keras.regularizers.L2(1e-4),
                       activity_regularizer=tf.keras.regularizers.L2(1e-5))(flatten)
        output = tf.keras.activations.sigmoid(output)
        self.model = Model(inputs=query_input, outputs=output)

    def call(self, inputs):
        return self.model(inputs)

    def build_graph(self, raw_shape):
        x = tf.keras.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class CNNATTENTION2D(Model):
    def __init__(self):
        super(CNNATTENTION2D, self).__init__()
        query_input = Input(shape=(14, 14, 1), dtype='float32')

        cnn_layer = Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu')(query_input)
        pool = MaxPooling2D(pool_size=(2, 2))(cnn_layer)
        norm = BatchNormalization()(pool)
        attention = Attention()([norm, norm])

        cnn_layer2 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(attention)
        pool2 = MaxPooling2D(pool_size=(2, 2))(cnn_layer2)
        norm2 = BatchNormalization()(pool2)
        attention2 = Attention()([norm2, norm2])

        cnn_layer3 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(attention2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(cnn_layer3)
        norm3 = BatchNormalization()(pool3)
        attention3 = Attention()([norm3, norm3])

        flatten = Flatten()(attention3)

        classification_output = Dense(2,
                                      kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                                      bias_regularizer=tf.keras.regularizers.L2(1e-4),
                                      activity_regularizer=tf.keras.regularizers.L2(1e-5))(flatten)
        output = Dense(2)(classification_output)
        self.model = Model(inputs=query_input, outputs=output)

    def call(self, inputs):
        return self.model(inputs)

    def build_graph(self, raw_shape):
        x = tf.keras.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = CNNATTENTION2D()
    cf.build_graph((14, 14, 1))
