from ele_common.units.networks import ConvLayer, DeConvLayer
from constants import *
from tensorflow import keras
from tensorflow.keras import layers


class Encoder(keras.Model):
    def __init__(self,
                 blocks,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 pool_size=(2, 1),
                 activation='relu',
                 *args,
                 **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.blocks = blocks
        self.block_size = len(blocks)

        for i, filters in enumerate(blocks):
            setattr(self, "conv_layer" + str(i), ConvLayer(
                filters=filters, kernel_size=kernel_size, strides=strides, activation=activation, pool_size=pool_size
            ))
            setattr(self, "dropout" + str(i), layers.GaussianDropout(0.3))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(self.block_size):
            x = getattr(self, "conv_layer" + str(i))(x)
            x = getattr(self, "dropout" + str(i))(x)

        return x


class Decoder(keras.Model):
    def __init__(self,
                 blocks,
                 kernel_size=(1, 3),
                 strides=(1, 1),
                 unpool_size=(1, 2),
                 activation='relu',
                 *args,
                 **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self.blocks = blocks
        self.block_size = len(blocks)

        for i, filters in enumerate(blocks):
            setattr(self, "de_conv_layer" + str(i), DeConvLayer(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=activation,
                unpool_size=unpool_size,
            ))
            setattr(self, "dropout" + str(i), layers.GaussianDropout(0.3))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(self.block_size):
            x = getattr(self, "de_conv_layer" + str(i))(x)
            x = getattr(self, "dropout" + str(i))(x)

        return x
