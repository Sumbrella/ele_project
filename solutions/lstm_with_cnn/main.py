import paddle
from paddle import fluid

from ele_common.units import generator
from ele_common.units.networks import ResNet50


def main():
    network = ResNet50()


if __name__ == '__main__':
    main()