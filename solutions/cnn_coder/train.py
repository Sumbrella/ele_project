import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from loguru import logger
from tensorflow import keras
from ele_common.units import SingleFile
from solutions.cnn_coder.model import CnnModel


def read_data(data_path, placeholder):
    logger.info("Logging data {}...".format(placeholder))

    sf = SingleFile(data_path)
    all_point_number = sf.point_number
    data = []
    # total参数设置进度条的总长度
    with tqdm(total=all_point_number) as pbar:
        pbar.set_description("Reading {}".format(placeholder))
        for i in range(all_point_number):
            data.append(sf.get_one_point().get_data())
            pbar.update(1)

    data = np.array(data).reshape(all_point_number, 1, -1, 2)

    logger.success("{} data read Done!".format(placeholder))

    return data


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
            output = np.tile(output, reps=[3, 1, 1])
            outputs.append(output.tolist())
            pbar.update(1)

    return np.array(outputs)


@logger.catch
def train(train_path, teacher_path):
    callback = keras.callbacks.TensorBoard(
        log_dir="./logs",
    )

    model = CnnModel(core_size=50)
    train_data = read_data(train_path, "train")
    train_data = handle_input(train_data, max_input_length=500)

    teacher_data = read_data(teacher_path, "teacher")
    teacher_data = handle_input(teacher_data, max_input_length=500)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer="adam",
    )

    model.fit(
        train_data,
        teacher_data,
        epochs=10,
        batch_size=10,
        callbacks=callback,
        validation_split=0.2,
    )


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "train_path",
    )

    parser.add_argument(
        "teacher_path",
    )

    parser.add_argument(
        "-epoches",
        default=10
    )

    parser.add_argument(
        ""
    )


if __name__ == '__main__':
    logger.add(
        open("./logs/loguru_log.log", "w+", encoding="utf-8")
    )
    train("../../data/generate/concat/data_result.dat", "../../data/generate/concat/teacher_result.dat")
    # train("../../data/origin/before/LINE_100_dbdt.dat", "../../data/origin/before/LINE_100_dbdt.dat")
