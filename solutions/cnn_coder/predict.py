import sys
sys.path.append("../..")

import numpy as np
from tensorflow.keras.models import load_model
from solutions.cnn_coder.train import data_handle
from ele_common.units import SingleFile
import matplotlib.pyplot as plt
from constants import *
from tensorflow.keras import backend as K

model = load_model("ed_model")
max_input_length = 500


tf = SingleFile("../../data/generate/concat/teacher_result.dat")
df = SingleFile("../../data/generate/concat/data_result.dat")

for i in range(10):
    tpoint = tf.get_one_point()
    point = df.get_one_point()
    tdata = data_handle(tpoint.x, tpoint.y, MAX_INPUT_LENGTH)
    ddata = data_handle(point.x, point.y, MAX_INPUT_LENGTH)

    tx = [t[0] for t in tdata[0]]
    ty = [t[1] for t in tdata[0]]

    dx = [t[0] for t in ddata[0]]
    dy = [t[1] for t in ddata[0]]
    x_data, y_data = point.get_data()

    output = data_handle(x_data, y_data, MAX_INPUT_LENGTH)
    data = []
    data.append(output.tolist())

    result = model(K.cast_to_floatx(np.array(data)))
    rx = [x[0] for x in result[0][0]]
    ry = [x[1] for x in result[0][0]]
    plt.scatter(tx, ty)
    plt.scatter(dx, dy)
    plt.scatter(rx, ry, alpha=0.5)
    plt.show()
