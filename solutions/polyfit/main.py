from paddle import fluid
from paddle.fluid.dygraph.nn import Conv2D, ParamAttr, MSELoss
from paddle.fluid.dygraph import BatchNorm, to_variable
from paddle.fluid.regularizer import L2Decay

from solutions.polyfit.get_reader import get_reader
from ele_common.units import Trainer, Reader


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



class DownSample(fluid.dygraph.Layer):
    """
    下采样，图片尺寸减半，具体实现方式是使用stirde=2的卷积
    """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=(1, 3),
                 stride=2,
                 padding=(0, 1),
                 is_test=True):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            is_test=is_test)
        self.ch_out = ch_out

    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(fluid.dygraph.Layer):
    """
    基本残差块的定义，输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
    """
    def __init__(self, ch_in, ch_out, is_test=True):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class LayerWarp(fluid.dygraph.Layer):
    """
    添加多层残差块，组成Darknet53网络的一个层级
    """
    def __init__(self, ch_in, ch_out, count, is_test=True):
        super(LayerWarp,self).__init__()

        self.basicblock0 = BasicBlock(ch_in,
            ch_out,
            is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i),
                BasicBlock(ch_out*2,
                    ch_out,
                    is_test=is_test))
            self.res_out_list.append(res_out)

    def forward(self,inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(fluid.dygraph.Layer):
    def __init__(self,

                 is_test=True):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(
            ch_in=2,
            ch_out=32,
            filter_size=(1, 3),
            stride=1,
            padding=1,
            is_test=is_test)

        # 下采样，使用stride=2的卷积来实现
        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2,
            is_test=is_test)

        # 添加各个层级的实现
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32 * (2 ** (i + 1)),
                          32 * (2 ** i),
                          stage,
                          is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)
        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32 * (2 ** (i + 1)),
                           ch_out=32 * (2 ** (i + 2)),
                           is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self, inputs):
        out = self.conv0(inputs)
        # print("conv1:",out.numpy())
        out = self.downsample0(out)
        print(out)
        # print("dy:",out.numpy())
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):  # 依次将各个层级作用在输入上面
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)

        return blocks[-1:-4:-1]  # 将C0, C1, C2作为返回值


elt_network_ctg = [2, 8, 4]


class EleNetwork(fluid.dygraph.Layer):
    def __init__(self, stages=None, is_test=True):
        super(EleNetwork, self).__init__()
        if stages is None:
            stages = elt_network_ctg

        self.stages = stages
        self.darknet53_conv_block_list = []
        self.downsample_list = []

        self.conv0 = ConvBNLayer(
            ch_in=2,
            ch_out=32,
            filter_size=(1, 3),
            stride=1,
            padding=(0, 1),
            is_test=is_test
        )

        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32*2,
            is_test=is_test
        )

        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32 * (2 ** (i + 1)),
                          32 * (2 ** i),
                          stage,
                          is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)

        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32 * (2 ** (i + 1)),
                           ch_out=32 * (2 ** (i + 2)),
                           is_test=is_test))
            self.downsample_list.append(downsample)

        self.fc1 = fluid.dygraph.Linear(input_dim=256*13, output_dim=9, act='relu')

        self.fc2 = fluid.dygraph.Linear(input_dim=9, output_dim=1, act='relu'
                                        )

    def forward(self, inputs):
        out = self.conv0(inputs)
        print('conv0_out', out)
        # print("conv1:",out.numpy())
        out = self.downsample0(out)

        # print("dy:",out.numpy())
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):  # 依次将各个层级作用在输入上面
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        # print(out)
        out = fluid.layers.reshape(x=out, shape=[-1, 256*13])
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc2(out)
        # print('out:', out)
        return out


if __name__ == '__main__':
    import numpy as np
    from ele_common.units import SingleFile

    train_dir = "../../data/train/before"
    train_label_dir = "../../data/train/teacher"
    test_dir = "../../data/test/before"
    test_label_dir = "../../data/test/teacher"

    train_reader = get_reader(train_dir, train_label_dir, batch_size=15)
    test_reader = get_reader(test_dir, test_label_dir, batch_size=15)

    # =====
    # tmp_data = np.random.randn(1, 2, 1, 100).astype('float64')
    # tmp_data = tmp_data / 10000000
    # with fluid.dygraph.guard():
    #     tmp_data = to_variable(tmp_data)
    #     net = EleNetwork()
    #     net(tmp_data)
    # =====
    
    lr = 0.1
    epoch_num = 10
    loss_function = MSELoss()

    train_losses = []
    # mode = 'debug'
    mode = 'train'


    with fluid.dygraph.guard():
        model = EleNetwork()
        # trainer = Trainer(name="ele", loss_function=fluid.layers.square_error_cost)
        reader = Reader(train=train_reader, test=test_reader)


        model.train()
        optimizer = fluid.optimizer.Adam(learning_rate=lr, parameter_list=model.parameters())

        for epoch in range(epoch_num):
            for batch, data in enumerate(reader.train()):
                imgs, labels = data
                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                # print(logits)
                loss = fluid.layers.square_error_cost(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                print(avg_loss.numpy())
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
                # print(labels)
                imgs = fluid.dygraph.to_variable(imgs)
                labels = fluid.dygraph.to_variable(labels)
                logits = model(imgs)
                print(logits)
                loss = fluid.layers.square_error_cost(logits, labels)
                avg_loss = fluid.layers.mean(loss)
                # accuracy = fluid.layers.accuracy(logits, labels)
                # accuracies.append(accuracy.numpy())


            print(f"epoch:{epoch} test_result: loss | {np.mean(avg_loss.numpy())}")
            model.train()

