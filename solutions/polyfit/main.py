import paddle
from paddle import fluid
from paddle.fluid.dygraph.nn import Linear, BatchNorm, Dropout


class Network(fluid.dygraph.Layer):

    def __init__(self, input_dim, out_dim=9):
        super(Network, self).__init__()

        self.fc1 = Linear(
            input_dim=input_dim,
            output_dim=50,
            param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(0, 0.02)),
            bias_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(0, 0.03)),
            act='relu'
        )

        self.fc2 = Linear(
            input_dim=50,
            output_dim=out_dim,
            param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(0, 0.02)),
            bias_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Normal(0, 0.03)),
            act='relu'
        )

        self.dropout1 = Dropout(p=0.5)
        self.dropout2 = Dropout(p=0.5)

    def forward(self, input):
        output = self.fc1(input)
        output = self.dropout1(output)
        output = self.fc2(output)

        return output


if __name__ == '__main__':
    from common.unit import SingleFile
    from solutions.polyfit.fit_one_point import fit_one_point

    before_data_path = "../../data/origin/before/LINE_100_dbdt.dat"
    after_data_path = "../../data/origin/after/new_LINE_100_dbdt.dat"

    before_file = SingleFile(before_data_path)
    after_data_path = SingleFile(after_data_path)

    with fluid.dygraph.guard():
         point = before_file.get_one_point()





