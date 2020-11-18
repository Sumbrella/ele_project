import matplotlib.pyplot as plt
import pandas as pd

from paddle import fluid
from paddle.fluid import ParamAttr, Pool2D, Conv2D, Linear, BatchNorm
from paddle.fluid.regularizer import L2Decay

from solutions.polyfit.get_reader import get_reader
from solutions.polyfit.teacher_change import change_teacher


class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size=(1, 3),
                 stride=1,
                 groups=1,
                 padding=(0, 1),
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0, 0.2)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=num_filters,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0, 0.2),
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


class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()

        self.conv1 = ConvBNLayer(num_channels=2, num_filters=96, filter_size=11, stride=4, padding=5, act='leaky_relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = ConvBNLayer(num_channels=96, num_filters=256, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = ConvBNLayer(num_channels=256, num_filters=384, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.conv4 = ConvBNLayer(num_channels=384, num_filters=384, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.conv5 = ConvBNLayer(num_channels=384, num_filters=256, filter_size=3, stride=1, padding=1, act='leaky_relu')
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = Linear(input_dim=256 * 3, output_dim=256 * 2, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=256 * 2, output_dim=256 * 1, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=256, output_dim=num_classes)

        self.conv_layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, self.conv5,
                            self.pool5]

    def forward(self, x):
        for layers in self.conv_layers:
            x = layers(x)

        # x = self.bn1(x)

        x = fluid.layers.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)
        x = fluid.layers.dropout(x, self.drop_ratio1)

        x = self.fc2(x)
        x = fluid.layers.dropout(x, self.drop_ratio2)

        x = self.fc3(x)

        return x


if __name__ == '__main__':
    import numpy as np
    from common.units import SingleFile, Reader

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    mode = 'train'
    epoch_num = 150
    loss_function = fluid.layers.square_error_cost
    batch_size = 10
    bd = [1000, 2000, 3000]
    value = [1.0, 0.5, 0.1, 0.05]
    # lr = fluid.layers.piecewise_decay(boundaries=[1000, 2000, 3000], values=[1.0, 0.5, 0.1, 0.05])
    lr = 1.0
    l2 = fluid.regularizer.L2Decay(regularization_coeff=1e-2)

    train_reader = get_reader(train_dir, train_label_dir, batch_size=batch_size)
    test_reader = get_reader(test_dir, test_label_dir, batch_size=batch_size)
    min_losses = 1000000

    # =====
    # tmp_data = np.random.randn(1, 2, 1, 100).astype('float64')
    # tmp_data = tmp_data / 10000000
    # with fluid.dygraph.guard():
    #     tmp_data = to_variable(tmp_data)
    #     net = EleNetwork()
    #     net(tmp_data)
    # =====
    train_losses = []
    test_losses = []

    with fluid.dygraph.guard():
        model = AlexNet()
        reader = Reader(train=train_reader, test=test_reader)

        model.train()
        optimizer = fluid.optimizer.Adam(learning_rate=0.1,
                                         parameter_list=model.parameters(),
                                         # regularization=l2,
                                         )

        for epoch in range(epoch_num):
            for batch, data in enumerate(reader.train()):
                imgs, labels = data
                # imgs = np.log(imgs)
                # change teacher label value
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                loss = loss_function(logits, labels)
                avg_loss = fluid.layers.mean(loss)

                if batch % 10 == 0:
                    train_losses.append(avg_loss.numpy())
                    print(f"epoch:{epoch} batch:{batch} loss:{avg_loss.numpy()}")

                if mode == 'debug':
                    print("label:", labels.numpy())
                    print("logits:", logits.numpy())
                    print("loss:", loss.numpy())
                    print("avg_loss:", avg_loss.numpy())

                avg_loss.backward()

                optimizer.minimize(avg_loss)

                model.clear_gradients()

            model.eval()

            losses = []

            for batch, data in enumerate(reader.test()):
                imgs, labels = data

                # change labels
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)

                labels = fluid.dygraph.to_variable(labels)

                logits = model(imgs)

                loss = loss_function(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                losses.append(avg_loss.numpy())

            if np.mean(losses) < min_losses:
                min_losses = np.mean(losses)
                fluid.save_dygraph(model.state_dict(), "min_polyfit")
                fluid.save_dygraph(optimizer.state_dict(), "min_polyfit")

            print(f"epoch:{epoch} test_result: loss | {np.mean(losses)}")

            model.train()

        fluid.save_dygraph(model.state_dict(), "polyfit")
        fluid.save_dygraph(optimizer.state_dict(), "polyfit")

    train_losses = pd.DataFrame(train_losses)
    train_losses.plot()
    plt.show()

