import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# from solutions.cnn_coder.coders import Encoder, Decoder
from solutions.cnn_coder.coders_v2 import Encoder, Decoder


class CnnModel(tf.keras.Model):
    def __init__(self,
                 output_shape_,
                 core_size=10,
                 max_input_length=500,
                 ):
        super(CnnModel, self).__init__()

        # # version 1
        # self.encoder = Encoder(core_size, max_input_length, name="encoder")
        # self.decoder = Decoder(core_size, max_input_length, name="decoder")

        # version 2
        self.en_blocks = [32, 32, 64, 64, 128]
        self.de_blocks = [128, 64, 64, 32, 32]

        self.encoder = Encoder(blocks=self.en_blocks, name="encoder", pool_size=(1, 2))
        self.decoder = Decoder(blocks=self.de_blocks, name="decoder", unpool_size=(1, 2))

        self.output_shape_ = output_shape_
        self.output_size = 1
        for i in output_shape_: self.output_size *= i
        self.flatten = layers.Flatten()
        self.out_layer = layers.Dense(
            units=self.output_size
        )

    def call(self, inputs, training=None, mask=None):
        outputs = self.encoder(inputs)
        outputs = self.decoder(outputs)
        outputs = self.flatten(outputs)
        outputs = self.out_layer(outputs)

        return tf.reshape(outputs, shape=[-1, *self.output_shape_])


def handle_input(inputs, max_input_length, type):
    # inputs: (N, H, W, C)
    outputs = []
    with tqdm(total=len(inputs)) as pbar:
        for input in inputs:
            # print(input.shape)
            pbar.set_description("Handling data...")
            length = input.shape[1]
            output = np.pad(input,
                            pad_width=((0, 0), (0, max_input_length - length), (0, 0)),
                            constant_values=(0, 0),
                            )
            if type == "input":
                output = np.tile(output, reps=[3, 1, 1])
            outputs.append(output.tolist())
            pbar.update(1)

    if type == "input":
        outputs = np.array(outputs).flatten().reshape(len(inputs), 3, -1, 2)

    elif type == "output":
        outputs = np.array(outputs).flatten().reshape(len(inputs), 1, -1, 2)

    else:
        raise ValueError(f"Unknown type {type}")

    return K.cast_to_floatx(outputs)


if __name__ == '__main__':
    import numpy as np
    from ele_common.units import SinglePoint

    callbacks = tf.keras.callbacks.TensorBoard("./logs")

    # test_x = np.random.randn(10, 1, 250, 2)
    # test_y = np.random.randn(10, 1, 250, 2)
    fp = open("../../data/origin/before/LINE_100_dbdt.dat", "r+")
    fp.readline()
    fp.readline()

    point = SinglePoint.from_file(fp)

    test_x = [point.get_data(), point.get_data()]
    test_x = np.array(test_x).reshape(2, 1, -1, 2)
    print(test_x)

    max_input_length = 512

    test_x, test_y = handle_input(test_x, max_input_length, "input"), handle_input(test_x, max_input_length, "output")

    print(test_x[0])

    model = CnnModel(output_shape_=(1, max_input_length, 2))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
    )

    model.fit(
        test_x,
        test_y,
        epochs=5,
        callbacks=callbacks,
    )
