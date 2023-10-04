import tensorflow as tf
from keras import layers, models


class SPConv2D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, alpha=0.5):
        super().__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha

        self.in_rep_channels = int(in_channels * self.alpha)
        self.out_rep_channels = int(out_channels * self.alpha)
        self.out_channels = out_channels
        self.stride = stride

        self.represent_gp_conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(3, 3),
            strides=self.stride,
            padding='same',
            groups=2
        )
        self.represent_pt_conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1
        )

        self.redundant_pt_conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1
        )

        self.avg_pool_s2_1 = tf.keras.layers.AveragePooling2D(
            pool_size=2,
            strides=2
        )
        self.avg_pool_s2_3 = tf.keras.layers.AveragePooling2D(
            pool_size=2,
            strides=2
        )

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.avg_pool_add_1 = tf.keras.layers.GlobalAvgPool2D()
        self.avg_pool_add_3 = tf.keras.layers.GlobalAvgPool2D()

        self.group = int(1 / self.alpha)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        x_3x3 = x[:, :, :, :self.in_rep_channels]
        x_1x1 = x[:, :, :, self.in_rep_channels:]
        rep_gp = self.represent_gp_conv(x_3x3)

        if self.stride == 2:
            x_3x3 = self.avg_pool_s2_3(x_3x3)
        rep_pt = self.represent_pt_conv(x_3x3)
        rep_fuse = rep_gp + rep_pt
        rep_fuse = self.bn1(rep_fuse)
        rep_fuse_ration = tf.reduce_mean(rep_fuse, axis=[1, 2])

        if self.stride == 2:
            x_1x1 = self.avg_pool_s2_1(x_1x1)
        red_pt = self.redundant_pt_conv(x_1x1)
        red_pt = self.bn2(red_pt)
        red_pt_ratio = tf.reduce_mean(red_pt, axis=[1, 2])

        out_31_ratio = tf.stack((rep_fuse_ration, red_pt_ratio), axis=1)
        out_31_ratio = tf.keras.activations.softmax(out_31_ratio, axis=1)

        temp1 = tf.reshape(out_31_ratio[:, 1, :],
                           shape=(batch_size, 1, 1, out_31_ratio[:, 1, :].shape[1]))
        out_mul_1 = red_pt * temp1

        temp2 = tf.reshape(out_31_ratio[:, 0, :],
                           shape=(batch_size, 1, 1, out_31_ratio[:, 0, :].shape[1]))
        out_mul_3 = rep_fuse * temp2
        return out_mul_1 + out_mul_3


# class SPConv1D(tf.keras.layers.Layer):
#     def __init__(self, in_channels, out_channels, stride=1, alpha=0.5):
#         super(SPConv1D, self).__init__()
#         assert 0 <= alpha <= 1
#         self.alpha = alpha
#
#         self.in_rep_channels = int(in_channels * self.alpha)
#         self.out_rep_channels = int(out_channels * self.alpha)
#         self.out_channels = out_channels
#         self.stride = stride
#
#         self.represent_gp_conv = tf.keras.layers.Conv1D(
#             filters=self.out_channels,
#             kernel_size=3,
#             strides=self.stride,
#             padding='same',
#             groups=2
#         )
#         self.represent_pt_conv = tf.keras.layers.Conv1D(
#             filters=self.out_channels,
#             kernel_size=1
#         )
#
#         self.redundant_pt_conv = tf.keras.layers.Conv1D(
#             filters=self.out_channels,
#             kernel_size=1
#         )
#
#         self.avg_pool_s2_1 = tf.keras.layers.AveragePooling1D(
#             pool_size=2,
#             strides=2
#         )
#         self.avg_pool_s2_3 = tf.keras.layers.AveragePooling1D(
#             pool_size=2,
#             strides=2
#         )
#
#         self.avg_pool_add_1 = tf.keras.layers.GlobalAveragePooling1D()
#         self.avg_pool_add_3 = tf.keras.layers.GlobalAveragePooling1D()
#
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.bn2 = tf.keras.layers.BatchNormalization()
#
#         self.group = int(1 / self.alpha)
#
#     def call(self, x):
#         batch_size = tf.shape(x)[0]
#
#         x_3x3 = x[:, :, :self.in_rep_channels]
#         x_1x1 = x[:, :, self.in_rep_channels:]
#         rep_gp = self.represent_gp_conv(x_3x3)
#
#         if self.stride == 2:
#             x_3x3 = self.avg_pool_s2_3(x_3x3)
#         rep_pt = self.represent_pt_conv(x_3x3)
#         rep_fuse = rep_gp + rep_pt
#         rep_fuse = self.bn1(rep_fuse)
#         rep_fuse_ration = self.avg_pool_add_3(rep_fuse)
#
#         if self.stride == 2:
#             x_1x1 = self.avg_pool_s2_1(x_1x1)
#
#         red_pt = self.redundant_pt_conv(x_1x1)
#         red_pt = self.bn2(red_pt)
#         red_pt_ratio = self.avg_pool_add_1(red_pt)
#
#         out_31_ratio = tf.stack((rep_fuse_ration, red_pt_ratio), axis=2)
#         out_31_ratio = tf.keras.activations.softmax(out_31_ratio, axis=2)
#
#         out_mul_1 = red_pt * tf.expand_dims(out_31_ratio[:, :, 1], axis=2)
#         out_mul_3 = rep_fuse * tf.expand_dims(out_31_ratio[:, :, 0], axis=2)
#
#         return out_mul_1 + out_mul_3


class SP(tf.keras.Model):
    def __init__(self, classification_mode, pretrained=None):
        super().__init__()
        if pretrained is not None:
            pass
        else:
            if classification_mode == 'multi':
                self.n_classes = 5
            elif classification_mode == 'binary':
                self.n_classes = 2
            self.activation = tf.keras.layers.Softmax(axis=1)
            self._model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    input_shape=(14, 14, 1),
                    filters=8,
                    kernel_size=2,
                    strides=1,
                    activation='relu',
                    padding='same'
                ),
                tf.keras.layers.ReLU(),
                SPConv2D(
                    in_channels=8,
                    out_channels=16,
                    stride=1,
                    alpha=0.8
                ),
                tf.keras.layers.ReLU(),
                SPConv2D(
                    in_channels=16,
                    out_channels=32,
                    stride=1,
                    alpha=0.8
                ),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Flatten(data_format='channels_last')
            ])
            self.fc = tf.keras.layers.Dense(units=self.n_classes)

    def call(self, x):
        x = self._model(x)
        x = self.fc(x)

        return x

    def build_graph(self, raw_shape):
        x = tf.keras.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    model = SP(classification_mode='multi')
    raw_input = (11, 11, 1)
    # y = model(tf.ones(shape=(0, *raw_input)))
    model.build_graph(raw_input).summary()
