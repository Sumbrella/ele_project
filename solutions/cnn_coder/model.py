import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from solutions.cnn_coder.coders import Encoder, Decoder


class CnnModel(tf.keras.Model):
    def __init__(self,
                 core_size=10,
                 max_input_length=500,
                 ):
        super(CnnModel, self).__init__()

        self.encoder = Encoder(core_size, max_input_length, name="encoder")
        self.decoder = Decoder(core_size, max_input_length, name="decoder")

    def call(self, inputs, training=None, mask=None):
        inputs = self.encoder(inputs)
        outputs = self.decoder(inputs)

        return outputs


# def handle_input(inputs, max_input_length):
#     # inputs: (N, H, W, C)
#     outputs = []
#     for input in inputs:
#         length = input.shape[2]
#         output = np.pad(input,
#                         pad_width=((0, 0), (0, max_input_length - length), (0, 0)),
#                         constant_values=(0, 0),
#                         )
#         output = np.tile(output, reps=[3, 1, 1])
#         outputs.append(output.tolist())
#     return outputs


def handle_input(inputs, max_input_length):
    # inputs: (N, H, W, C)
    outputs = []
    with tqdm(total=len(inputs)) as pbar:
        for input in inputs:
            pbar.set_description("Handling data...")
            length = input.shape[2]
            output = np.pad(input,
                            pad_width=((0, 0), (0, max_input_length - length), (0, 0)),
                            constant_values=(0, 0),
                            )
            # input = output
            output = np.tile(output, reps=[3, 1, 1])
            outputs.append(output.tolist())
            # outputs = np.vstack([outputs, output])
            pbar.update(1)
    outputs = np.array(outputs).flatten().reshape(len(inputs), 1, -1, 2)
    # return outputs
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

    test_x = handle_input(test_x, 500)
    test_y = test_x

    print(test_x[0])

    model = CnnModel()

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
