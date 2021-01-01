import tensorflow.keras.models as models
from tensorflow.keras.layers import Layer, Activation, Reshape, Permute
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np

np.random.seed(123123) # 0bserver07 for reproducibility

data_shape = (3, 500)


def create_encoding_layers():
    kernel = (1, 2)
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, 'valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, 'valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, 'valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, 'valid'),
        BatchNormalization(),
        Activation('relu'),
    ]


def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, 'valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, 'valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, 'valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, 'valid'),
        BatchNormalization(),
    ]


segnet_basic = models.Sequential()

segnet_basic.add(Layer(input_shape=(3, 360, 480)))

segnet_basic.encoding_layers = create_encoding_layers()

for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)

# Note: it this looks weird, that is because of adding Each Layer using that for loop
# instead of re-writting mode.add(somelayer+params) everytime.

segnet_basic.decoding_layers = create_decoding_layers()
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)

segnet_basic.add(Convolution2D(12, 1, 1, 'valid',))

segnet_basic.add(Reshape((12, data_shape), input_shape=(12, *data_shape)))
segnet_basic.add(Permute((2, 1)))
# segnet_basic.add(Activation('softmax'))
