from loguru import logger


class Checker:
    def __init__(self):
        pass

    @classmethod
    def check(cls, x, y, xsorted=True, debug=False):
        """
        输入的x, y是否单调递减。
        如果x不按照升序排列，需要关闭xsorted
        :param x: 横坐标
        :param y: 纵坐标
        :param xsorted: x是否按升序排列
        :param debug:  是否 debug
        :return: bool: 检查结果
        :type x: list
        :type y: list
        :type xsorted: bool
        :type debug: bool
        :rtype: bool
        """

        # 检查长度
        n = len(x)
        m = len(y)

        if n != m:
            logger.error('In module Checker.check, the x data and y data should have the same size'
                         'but x:{} | y:{}'.format(n, m))
            raise ValueError('shape can not be difference')

        if debug:
            import matplotlib.pyplot as plt
            plt.plot(x, y)
            plt.show()

        # 横坐标升序排序
        if not xsorted:
            x, y = sorted(zip(x, y))

        # 检查递增性
        for i in range(n - 1):
            if y[i + 1] > y[i]:
                return False
        return True

