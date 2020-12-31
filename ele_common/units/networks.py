from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
    "ResNet_v2"
]


class ResNet:
    def __init__(self, layers=50):
        self.layers = layers

    def net(self, input, class_dim=1000, data_format="NCHW"):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max',
            data_format=data_format)
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name,
                        data_format=data_format)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True, data_format=data_format)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name,
                        data_format=data_format)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True, data_format=data_format)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format='NCHW'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
            data_format=data_format)

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        if data_format == 'NCHW':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[-1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b",
            data_format=data_format)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1",
            data_format=data_format)

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basic_block(self, input, num_filters, stride, is_first, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b",
            data_format=data_format)
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1", data_format=data_format)
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


def ResNet18():
    model = ResNet(layers=18)
    return model


def ResNet34():
    model = ResNet(layers=34)
    return model


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model


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
