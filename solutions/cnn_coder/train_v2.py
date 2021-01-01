import numpy as np
# from argparse import ArgumentParser
from tqdm import tqdm
from loguru import logger
from tensorflow import keras
from ele_common.units import SingleFile
# from solutions.cnn_coder.model import CnnModel
from tensorflow.keras import backend as K
from ele_common.units.seg_basic import segnet_basic as model


def data_handle(x):
    return abs(np.log10(x) / 4)


def read_data(data_path, placeholder, max_input_length=500):
    logger.info("Logging data {}...".format(placeholder))

    sf = SingleFile(data_path)
    data = []
    all_point_number = 50
    # all_point_number = sf.point_number
    with tqdm(total=all_point_number) as pbar:
        pbar.set_description("Reading {}".format(placeholder))
        for i in range(all_point_number):
            point = sf.get_one_point()
            one_data = point.get_data()
            one_data = np.array(one_data).reshape(1, -1, 2)
            one_data = data_handle(one_data)
            length = point.size
            output = np.pad(one_data,
                            pad_width=((0, 0), (0, max_input_length - length), (0, 0)),
                            constant_values=(0, 0),
                            )
            output = np.tile(output, reps=[3, 1, 1])
            data.append(output.tolist())
            pbar.update(1)

    return K.cast_to_floatx(np.array(data))


@logger.catch
def train(train_path, teacher_path, core_size, epochs, batch_size, max_input_length):
    callback = keras.callbacks.TensorBoard(
        log_dir="./logs",
    )

    # model = CnnModel(core_size=core_size, max_input_length=max_input_length)
    train_data = read_data(train_path, "train")
    teacher_data = read_data(teacher_path, "teacher")

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer="adam",
    )

    model.fit(
        train_data,
        teacher_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback,
        validation_split=0.2,
        validation_steps=1,
    )

    model.save("ed_model", include_optimizer=True)


if __name__ == '__main__':
    train("../../data/generate/concat/data_result.dat", "../../data/generate/concat/teacher_result.dat")