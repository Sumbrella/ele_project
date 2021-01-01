import tensorflow as tf
from tensorflow.keras import layers
from ele_common.units.networks import Block, ResNet_v2


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 activation='relu',
                 *args,
                 **kwargs
                 ):
        super(ConvLayer, self).__init__(*args, **kwargs)

        self.batch_norm = layers.BatchNormalization()

        self.conv_1 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

        self.conv_2 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

        self.conv_3 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation='relu'
        )

        self.avg_pool = layers.AveragePooling2D(
            pool_size=2,
        )

    def call(self, inputs, training=None):
        outputs = self.batch_norm(inputs)
        outputs = self.conv_1(outputs)
        outputs = self.conv_2(outputs)
        outputs = self.conv_3(outputs)
        outputs = self.avg_pool(outputs)

        return outputs


class DeConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 activation='relu',
                 *args,
                 **kwargs,
                 ):
        super(DeConvLayer, self).__init__(*args, **kwargs)

        self.batch_norm = layers.BatchNormalization()

        self.conv_1 = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation
        )

        self.conv_2 = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

        self.conv_3 = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

        self.unsample = layers.UpSampling2D(size=2)

    def call(self, inputs, training=None):
        outputs = self.batch_norm(inputs)
        outputs = self.conv_1(outputs)
        outputs = self.conv_2(outputs)
        outputs = self.conv_3(outputs)
        outputs = self.un_sample(outputs)

        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 core_size=10,
                 max_input_length=500,
                 *args,
                 **kwargs,
                 ):
        super(Encoder, self).__init__(*args, **kwargs)

        self.max_input_length = max_input_length

        self.conv_1 = layers.Conv2D(
            filters=32,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation='relu'
        )

        self.max_pool_1 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))

        self.conv_2 = layers.Conv2D(
            filters=32,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation='relu'
        )

        self.max_pool_2 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))

        self.conv_3 = layers.Conv2D(
            filters=64,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation=None,
        )

        self.dropout = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

        self.fc = layers.Dense(units=core_size)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv_1(inputs)
        # outputs = self.max_pool_1(outputs)
        outputs = self.conv_2(outputs)
        # outputs = self.max_pool_2(outputs)
        outputs = self.conv_3(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)

        return outputs


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self,
#                  core_size=10,
#                  max_input_length=500,
#                  *args, **kwargs,
#                  ):
#         super(Encoder, self).__init__(*args, **kwargs)
#         blocks_50 = [
#             Block("block1", [(256, 64, 1), (256, 64, 1), (256, 64, 2)]),
#             Block("block2", [(512, 128, 1)] * 3 + [(512, 128, 2)]),
#             Block("block3", [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
#             Block("block4", [(2048, 512, 1)] * 3)
#         ]
#         self.max_input_length = max_input_length
#         self.res_net = ResNet_v2(blocks_50)
#         self.fc = layers.Dense(units=core_size)
#
#     def call(self, inputs, training=None):
#         outputs = self.res_net(inputs)
#         outputs = self.fc(outputs)
#
#         return outputs


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 core_size=10,
                 max_input_length=500,
                 *args,
                 **kwargs
                 ):
        super(Decoder, self).__init__(*args, **kwargs)

        self.core_size = core_size
        self.max_input_length = max_input_length

        self.conv_3 = layers.Conv2DTranspose(
            filters=64,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation=None,
        )

        self.pool_2 = layers.UpSampling2D(size=(2, 2))

        self.conv_2 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation='relu'
        )

        self.pool_1 = layers.UpSampling2D(size=(2, 2))

        self.conv_1 = layers.Conv2DTranspose(
            filters=32,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation='relu'
        )

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=(1, 2),
            strides=1,
            padding='SAME',
            activation='relu'
        )

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv_3(inputs)
        # outputs = self.pool_2(outputs)
        outputs = self.conv_2(outputs)
        # outputs = self.pool_1(outputs)
        outputs = self.conv_1(outputs)
        outputs = self.conv(outputs)

        return outputs


if __name__ == '__main__':
    import numpy as np

    batch = 1000

    test_data = np.random.randn(batch, 1, 250, 2)

    encoder = Encoder()
    decoder = Decoder()
    cores = encoder(test_data)
    res = decoder(cores)
    print(res.shape)
