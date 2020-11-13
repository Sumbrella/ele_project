import matplotlib.pyplot as plt
import pandas as pd

from paddle import fluid
from paddle.fluid import Pool2D, Conv2D, Linear, BatchNorm

from solutions.polyfit.get_reader import get_reader
from solutions.polyfit.teacher_change import change_teacher


class AlexNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()

        self.bn1 = BatchNorm(num_channels=2, act='relu')

        self.conv1 = Conv2D(num_channels=2, num_filters=96, filter_size=11, stride=4, padding=5, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=96, num_filters=256, filter_size=5, stride=1, padding=2, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv3 = Conv2D(num_channels=256, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv4 = Conv2D(num_channels=384, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv5 = Conv2D(num_channels=384, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        self.pool5 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')

        self.fc1 = Linear(input_dim=256*3, output_dim=256*2, act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=256*2, output_dim=256*1, act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=256, output_dim=num_classes)

        self.conv_layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4, self.conv5, self.pool5]

    def forward(self, x):

        for layers in self.conv_layers:
            x = layers(x)
        # print(x.shape)
        x = fluid.layers.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)
        x = fluid.layers.dropout(x, self.drop_ratio1)

        x = self.fc2(x)
        x = fluid.layers.dropout(x, self.drop_ratio2)

        x = self.fc3(x)

        return x


if __name__ == '__main__':
    import numpy as np
    from common.unit import SingleFile, Reader

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    train_reader = get_reader(train_dir, train_label_dir, batch_size=10)
    test_reader = get_reader(test_dir, test_label_dir, batch_size=10)

    # =====
    # tmp_data = np.random.randn(1, 2, 1, 100).astype('float64')
    # tmp_data = tmp_data / 10000000
    # with fluid.dygraph.guard():
    #     tmp_data = to_variable(tmp_data)
    #     net = EleNetwork()
    #     net(tmp_data)
    # =====

    lr = 0.1
    epoch_num = 100
    loss_function = fluid.layers.cross_entropy

    train_losses = []
    # mode = 'debug'
    mode = 'train'

    with fluid.dygraph.guard():
        model = AlexNet()
        # trainer = Trainer(name="ele", loss_function=fluid.layers.square_error_cost)
        reader = Reader(train=train_reader, test=test_reader)

        model.train()
        optimizer = fluid.optimizer.Adam(learning_rate=lr, parameter_list=model.parameters())

        for epoch in range(epoch_num):
            for batch, data in enumerate(reader.train()):
                imgs, labels = data

                # change teacher label value
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                loss = fluid.layers.square_error_cost(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                # print(avg_loss.numpy())
                # train_losses.append(avg_loss)

                if mode == 'debug':
                    print("label:", labels.numpy())
                    print("logits:", logits.numpy())
                    print("loss:", loss.numpy())
                    print("avg_loss:", avg_loss.numpy())

                avg_loss.backward()

                optimizer.minimize(avg_loss)

                model.clear_gradients()

                if batch % 10 == 0:
                    train_losses.append(avg_loss.numpy())
                    print(f"epoch:{epoch} batch:{batch} loss:{avg_loss.numpy()}")

            model.eval()
            losses = []
            # accuracies = []
            for batch, data in enumerate(reader.test()):
                imgs, labels = data

                # change labels
                labels = change_teacher(labels)

                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                # print("label\n", labels)
                # print("logits\n", logits)
                loss = fluid.layers.square_error_cost(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                # accuracy = fluid.layers.accuracy(logits, labels)
                # accuracies.append(accuracy.numpy())

            print(f"epoch:{epoch} test_result: loss | {np.mean(avg_loss.numpy())}")

            model.train()

    train_losses = pd.DataFrame(train_losses)
    train_losses.plot()
    plt.show()
