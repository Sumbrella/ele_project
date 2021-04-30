import tensorflow as tf
import tensorboard
from tensorflow import keras
from tensorflow.keras import layers


class Changer(keras.Model):
    def __init__(self, max_length=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def call(self, inputs, training=None, mask=None):
        pass
