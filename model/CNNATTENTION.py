from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Attention, Flatten, Dense
from keras.models import Model


class CNNATTENTION(Model):
    def __init__(self):
        super(CNNATTENTION, self).__init__()
        query_input = Input(shape=(121, 1), dtype='float32')
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
        output = Dense(5)(flatten)
        self.model = Model(inputs=query_input, outputs=output)

    def call(self, inputs):
        return self.model(inputs)
