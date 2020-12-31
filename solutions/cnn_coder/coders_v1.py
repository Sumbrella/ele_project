import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Block(collections.namedtuple("block", ["name", "args"])):
    "A named tuple describing a ResNet Block"
    # collections.namedtuple() 函数原型为：
    # namedtuple(typename, field_names, verbose, renmae)


# 残差单元
class ResidualUnit(layers.Layer):
    def __init__(self, depth, depth_residual, stride):
        super(ResidualUnit, self).__init__()
        self.depth = depth
        self.depth_residual = depth_residual
        self.stride = stride

    def build(self, input_shape):
        self.depth_input = input_shape[-1]
        # layers.BatchNormalization归一化
        self.batch_normal = layers.BatchNormalization()

        self.identity_maxpool2d = layers.MaxPool2D(
            pool_size=(1, 1),
            strides=self.stride
        )

        self.identity_conv2d = layers.Conv2D(
            filters=self.depth,
            kernel_size=[1, 1],
            strides=self.stride,
            activation=None
        )

        self.conv1 = layers.Conv2D(
            filters=self.depth_residual,
            kernel_size=[1, 1],
            strides=1,
            activation=None,
        )

        self.conv_same = layers.Conv2D(
            filters=self.depth_residual,
            kernel_size=[3, 3],
            strides=self.stride,
            padding='SAME',
        )

        self.conv_valid = layers.Conv2D(
            filters=self.depth_residual,
            kernel_size=[3, 3],
            strides=self.stride,
            padding='VALID'
        )

        self.conv3 = layers.Conv2D(
            filters=self.depth,
            kernel_size=[1, 1],
            strides=1,
            activation=None
        )

    def call(self, inputs, training=None):
        # 使用BatchNormalizatino 进行归一化
        batch_norm = self.batch_normal(inputs)
        # 如果本块的depth值等于上一个块的depth值，考虑进行降采用操作
        # 如果不等于，则使用conv2d() 调整输入输出通道

        if self.depth == self.depth_input:
            # 如果 stride = 1 不进行降采样
            # 如果 steide != 1 使用max_pool2d进行步长为stride
            # 且池化核为1 * 1的降采用
            if self.stride == 1:
                identity = inputs
            else:
                identity = self.identity_maxpool2d(inputs)
        else:
            identity = self.identity_conv2d(batch_norm)

        # 一个残差块中三个卷积层的第一个卷积层
        residual = self.conv1(batch_norm)

        # 第二个卷积层
        # 如果 stride == 1 那么使用 SAME
        # 否则使用 "VALID"
        if self.stride == 1:
            residual = self.conv_same(residual)
        else:
            pad_begin = (3 - 1) // 2
            pad_end = 3 - 1 - pad_begin
            # pad() 函数用于对矩阵进行定制填充
            residual = tf.pad(
                residual,
                [
                    [0, 0], [pad_begin, pad_end],
                    [pad_begin, pad_end], [0, 0],
                ]
            )
            residual = self.conv_valid(residual)

        # 第三个卷积层
        residual = self.conv3(residual)

        return identity + residual


class ResNet_v2(tf.keras.Model):
    def __init__(self, blocks):
        super(ResNet_v2, self).__init__()
        self.blocks = blocks

        self.conv1 = layers.Conv2D(filters=64, kernel_size=[2, 2], strides=1)
        # self.pool1 = layers.MaxPool2D(pool_size=[3, 3], strides=2)

        for block in self.blocks:
            for i, tuple_value in enumerate(block.args):
                # 每一个tuple_valeu 由三个数组成
                depth, depth_residual, stride = tuple_value

                setattr(
                    self,
                    block.name + "_" + str(i + 1),
                    ResidualUnit(depth, depth_residual, stride)
                )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        # x = self.pool1(x)
        for block in self.blocks:
            for i, tuple_value in enumerate(block.args):
                # getattr 返回一个属性
                residual_unit = \
                    getattr(self, block.name + "_" + str(i + 1))
                x = residual_unit(x)

        return x


class FcNet(tf.keras.Model):
    def __init__(self,
                 core_shape,
                 *args,
                 **kwargs
                 ):
        super(FcNet, self).__init__(*args, **kwargs)

        self.core_shape = core_shape
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.fc = layers.Dense(core_shape)

    def call(self, inputs, training=None, mask=None):

        outputs = self.flatten(inputs)
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)

        return outputs


class Encoder(tf.keras.Model):
    def __init__(self,
                 blocks,
                 core_size=10,
                 max_input_length=500,
                 *args,
                 **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.res_net = ResNet_v2(blocks=blocks)
        self.fc_net = FcNet(core_size)

        self.core_size = core_size
        self.max_input_length = max_input_length

    def call(self, inputs, training=None, mask=None):

        inputs = self._handle_input(inputs)

        outputs = self.res_net(inputs)
        outputs = self.fc_net(outputs)

        return outputs

    def _handle_input(self, inputs):
        # inputs: (N, H, W, C)
        length = inputs.shape[2]
        outputs = tf.pad(inputs,
                         paddings=((0, 0), (0, 0), (0, self.max_input_length - length), (0, 0)),
                         constant_values=(0, 0),
                         )
        outputs = tf.tile(inputs, multiples=[1, 3, 1, 1])
        return outputs


if __name__ == '__main__':
    blocks_50 = [
        Block("block1", [(256, 64, 1), (256, 64, 1), (256, 64, 2)]),
        Block("block2", [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block("block3", [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block("block4", [(2048, 512, 1)] * 3)
    ]

    batch = 10
    core_size = 10

    net = Encoder(blocks=blocks_50, core_size=core_size)

    test_data = np.random.randn(10, 1, 250, 2)
    test_y = np.random.randn(batch, core_size)

    net.compile(
        loss=tf.keras.losses.MeanSquaredError(),
    )

    net.fit(
        x=test_data,
        y=test_y,
    )

    tf.keras.layers.DenseFeatures(

    )
