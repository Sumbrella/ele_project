import numpy as np


def log_change(x):
    return abs(np.log10(x) / 10)


def data_handle(x_data, y_data, max_input_length, func=log_change, out_shape=[1, -1, 2]):
    """
    this function will pad the data to the length of `max_input_length`
    and then use `func` function to apply to all elements in data.
    finally change the data to the `out_shape`.
    :return:
        numpy.array
    """
    data = np.array(list(zip(x_data, y_data))).reshape(out_shape)
    data = func(data)
    length = len(data[0])
    output = np.pad(
        data,
        pad_width=((0, 0), (0, max_input_length - length), (0, 0)),
        constant_values=(0, 0),
    ).reshape(out_shape)

    return output
