import numpy as np

from paddle import fluid
from paddle.fluid.dygraph import Linear, BatchNorm
from paddle.fluid.dygraph import Layer, MSELoss, NaturalExpDecay, to_variable

from ele_common.units import Reader
from solutions.polyfit.get_reader import get_reader


class OneHiddenNetwork(Layer):

    def __init__(self):
        super(OneHiddenNetwork, self).__init__()

        # self.bn1 = BatchNorm(num_channels=2, act='relu')

        self.fc1 = Linear(input_dim=28*28, output_dim=50, act='relu')
        self.fc2 = Linear(input_dim=50, output_dim=10, act='relu')

    def forward(self, input):

        # out = self.bn1(input)
        out = fluid.layers.reshape(input, shape=[input.shape[0], -1])
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def test_sample():
    tmp_data = np.random.randn(10, 28, 28).astype('float32')

    with fluid.dygraph.guard():
        tmp_data = fluid.dygraph.to_variable(tmp_data)
        net = OneHiddenNetwork()
        print(net(tmp_data))


def train():
    EPOCH_NUM = 10

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    train_reader = get_reader(train_dir, train_label_dir)
    test_reader = get_reader(test_dir, test_label_dir)

    # reader = Reader(train_reader, test_reader)
    reader = Reader.from_paddle('mnist')
    loss_function = MSELoss(reduction='sum')

    with fluid.dygraph.guard():
        model = OneHiddenNetwork()
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

                # loss = loss_function(logits, labels)
                loss = fluid.layers.cross_entropy(logits, labels)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                optimizer.minimize(avg_loss)

                model.clear_gradients()

                if batch % 10 == 0:
                    print(f'[INFO] batch: {batch}, loss: {avg_loss.numpy()}')


if __name__ == '__main__':
    test_sample()
    train()
    # from ele_common.units import Reader, Trainer
    #
    # reader = Reader.from_paddle(data_name='mnist')
    # trainer = Trainer(name="mnist")
    #
    # model = OneHiddenNetwork()
    # trainer.train(model, reader)
