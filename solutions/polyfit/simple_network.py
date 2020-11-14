import numpy as np
from paddle import fluid
from paddle.fluid.dygraph import Conv2D, Layer, NaturalExpDecay, BatchNorm, Linear, MSELoss
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph import to_variable
from paddle.fluid.regularizer import L2Decay

from solutions.polyfit.main import DownSample
from solutions.polyfit.get_reader import get_reader
from common.unit import Reader


class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=(1, 3),
                 stride=1,
                 groups=1,
                 padding=(0, 1),
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(100, 2)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(100, 2),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(100),
                regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out


class SimpleNetwork(Layer):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

        self.conv1 = ConvBNLayer(ch_in=2, ch_out=10, filter_size=(1, 3), stride=1, is_test=False)
        self.downsample1 = DownSample(ch_in=10, ch_out=10 * 2, is_test=False)
        self.conv2 = ConvBNLayer(ch_in=20, ch_out=1, is_test=False)
        self.downsample2 = DownSample(ch_in=1, ch_out=2, is_test=False)

        self.fc1 = Linear(input_dim=2 * 25, output_dim=9, act='relu')

    def forward(self, input):
        out = self.conv1(input)
        # print(out.shape)
        out = self.downsample1(out)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.downsample2(out)
        # print(out.shape)

        out = fluid.layers.reshape(x=out, shape=[-1, 50])
        # print(out.shape)
        out = self.fc1(out)
        # print(out.shape)
        return out


def test_sample():
    tmp_data = np.random.randn(1, 2, 1, 100).astype('float32')

    with fluid.dygraph.guard():
        tmp_data = fluid.dygraph.to_variable(tmp_data)
        net = SimpleNetwork()
        net(tmp_data)


def train():
    EPOCH_NUM = 10

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    train_reader = get_reader(train_dir, train_label_dir)
    test_reader = get_reader(test_dir, test_label_dir)

    reader = Reader(train_reader, test_reader)
    loss_function = MSELoss(reduction='sum')

    with fluid.dygraph.guard():
        model = SimpleNetwork()
        optimizer = fluid.optimizer.Adam(
            learning_rate=NaturalExpDecay(learning_rate=1, decay_steps=10, decay_rate=1),
            parameter_list=model.parameters(),
        )

        for epoch in range(EPOCH_NUM):
            for batch, data in enumerate(reader.train()):
                points, labels = data

                points = to_variable(points)
                labels = to_variable(labels)

                logits = model(points)

                loss = loss_function(logits, labels)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                optimizer.minimize(avg_loss)

                model.clear_gradients()

                if batch % 10 == 0:
                    print(f'[INFO] batch: {batch}, loss: {avg_loss.numpy()}')

            # TODO: test for each epoch


class hidden_network(fluid.dygraph.Layer):
    def __int__(self):
        super(hidden_network, self).__init__()


if __name__ == '__main__':
    # test_sample()
    train()
