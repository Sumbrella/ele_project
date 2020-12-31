import csv
import os
import empymod
import numpy as np
from math import log
from time import sleep
from loguru import logger
from gooey import Gooey
from gooey import GooeyParser
from scipy.constants import mu_0


default_layer_scope = [2, 5]

default_res_scope = [10, 400]

default_deep_scope = [5, 300]

default_square_scope = [1, 5]

default_time_scope = [200, 300]  # [150, 300]
default_time_range = [1e-4, 1e-2]

default_perturbation_rate = 80


def loop_tem1d(time, L_square, depth, res, verb_flag=0):

    # === GET REQUIRED FREQUENCIES ===
    time, freq, ft, ftarg = empymod.utils.check_time(
        time=time,          # Required times
        signal=-1,          # Switch-on response
        ft='dlf',           # Use DLF
        ftarg={'dlf': 'key_201_CosSin_2012'},  # Short, fast filter; if you
        # need higher accuracy choose a longer filter.
        verb=verb_flag,
    )

    # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    EM = empymod.model.bipole(
        # El. bipole source; half of one side.
        src=[L_square/2., L_square/2., 0, L_square/2., 0, 0],
        rec=[0, 0, 0, 0, 90],         # Receiver at the origin, vertical.
        depth=np.r_[0, depth],        # Depth-model, adding air-interface.
        res=np.r_[2e14, res],         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=True,                    # It is an el. source, but a magn. rec.
        strength=8,                   # To account for 4 sides of square loop.
        # Approx. the finite dip. with 3 points.
        srcpts=100,
        verb=verb_flag,
        # htarg={'dlf': 'key_101_2009'},  # Short filter, so fast.
    )

    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    EM_b = EM*mu_0
    EM_db = EM*2j*np.pi*freq*mu_0

    # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
    # Note: Here we just apply one filter. But it seems that square_tem1d can apply
    #       two filters, one before and one after the so-called front gate
    #       (which might be related to ``delay_rst``, I am not sure about that
    #       part.)
    cutofffreq = 4.5e10
    h = (1+1j*freq/cutofffreq)**-1
    h *= (1+1j*freq/3e5)**-1
    EM_b *= h
    EM_db *= h

    # === CONVERT TO TIME DOMAIN ===
    delay_rst = 1.8e-7
    EM_b, _ = empymod.model.tem(EM_b[:, None], np.array([1]),
                                freq, time+delay_rst, -1, ft, ftarg)
    EM_b = np.squeeze(EM_b)
    EM_db, _ = empymod.model.tem(EM_db[:, None], np.array([1]),
                                 freq, time+delay_rst, -1, ft, ftarg)
    EM_db = np.squeeze(EM_db)

    # === APPLY WAVEFORM ===
    return EM_b, EM_db


def input_choice(continue_option):
    option = input()
    if option == continue_option:
        return True
    return False


def is_dir_exist(path):
    if not os.path.exists(path):
        print()
        return False
    return True


def make_dir_with_input(path):
    print('[INPUT] Create {}?(y/n)>>>\n'.format(path))

    if input_choice('y'):
        logger.info('Making {} ...'.format(dir))
        os.makedirs(path)
        logger.info('succeed!')

    else:
        raise FileExistsError('{} do not exists'.format(path))


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
            logger.error('In module Checker.check, the x teacher and y teacher should have the same size'
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



class EleData:
    def __init__(self, data_dict):

        self.params = [
            'origin_data',
            'result_data',
            'time',
            'layer_number',
            'depths',
            'res',
        ]

        self.origin_data = data_dict['origin_data']
        self.result_data = data_dict['result_data']
        self.time = data_dict['time']
        self.layer_number = data_dict['layer_number']
        self.depths = data_dict['depths']
        self.res = data_dict['res']

    @staticmethod
    def _dir_init(save_dir):
        # 用于保存教师数据
        teacher_dir = os.path.join(save_dir, 'data')
        # 用于保存扰动数据
        data_dir = os.path.join(save_dir, 'teacher')
        # 用于保存其他标签
        label_dir = os.path.join(save_dir, 'label')

        if not is_dir_exist(save_dir):
            make_dir_with_input(path=save_dir)

        if not is_dir_exist(teacher_dir):
            os.mkdir(teacher_dir)

        if not is_dir_exist(data_dir):
            os.mkdir(data_dir)

        if not is_dir_exist(label_dir):
            os.mkdir(label_dir)

    def _add_point_to_file(self, full_path, which):

        if which not in ['origin', 'result']:
            logger.error('param \'which\' should be \'origin \' or \' result \'')
            raise ValueError('param \'which\' should be \'origin \' or \' result \'')

        save_data = None

        if which == 'origin':
            save_data = self.origin_data
        elif which == 'result':
            save_data = self.result_data

        if not os.path.exists(full_path):
            import time
            # 如果不存在文件，则创建文件并写入 0 和 日期
            fp = open(os.path.join(full_path), 'w')
            fp.write(' 0\n')
            fp.write(" ")
            fp.write(time.asctime())
            fp.write("\n")

        with open(os.path.join(full_path), 'r+') as fp:
            # 读取第一行的point数目，并令其+1
            point_number = int(fp.readline())
            fp.seek(0)
            fp.writelines(' {}\n'.format(point_number + 1))

        with open(os.path.join(full_path), 'a+') as fp:
            # 写入一个point
            fp.write(' {}\n'.format(point_number + 1))
            fp.write(' 1   2   3   4\n')
            fp.write(' xy\n')
            fp.write(' {}\n'.format(len(self.time)))

            for x, y in zip(self.time, save_data):
                fp.write(' {:e}\t\t{:e}\n'.format(x, y))

    def _add_point_label(self, full_path):
        with open(full_path, 'a+') as cv:
            fieldnames = ['layer_number', 'depths', 'res']
            writer = csv.DictWriter(cv, fieldnames=fieldnames)
            writer.writerow({
                'layer_number': self.layer_number,
                'depths': self.depths,
                'res': self.res,
            })

    def add_to_dat(self, save_dir, file_name):
        self._dir_init(save_dir)

        teacher_dir = os.path.join(save_dir, 'teacher')
        data_dir = os.path.join(save_dir, 'data')
        label_dir = os.path.join(save_dir, 'label')

        self._add_point_to_file(os.path.join(data_dir, file_name + '.dat'), which='origin')
        self._add_point_to_file(os.path.join(teacher_dir, 'NEW_' + file_name + '.dat'), which='result')
        self._add_point_label(os.path.join(label_dir, file_name + '.csv'))

    def draw(self):
        import matplotlib.pyplot as plt
        plt.plot(
            self.time,
            self.origin_data,
            alpha=1,
            label='origin',
        )
        plt.plot(
            self.time,
            self.result_data,
            alpha=0.7,
            ls='--',
            label="result",
        )
        plt.legend()
        plt.show()


class Generator:
    def __init__(self,
                 layer_number_scope=None,
                 depth_scope=None,
                 resistant_scope=None,
                 time_sequence_number=None,
                 time_scope=None,
                 square_scope=None,
                 perturbation_rate=None,
                 ):
        self.layer_number_scope = layer_number_scope or default_layer_scope
        self.depth_scope = depth_scope or default_deep_scope
        self.resistant_scope = resistant_scope or default_res_scope
        self.time_scope = time_scope or default_time_range
        self.time_sequence_number = time_sequence_number or default_time_scope
        self.perturbation_rate = perturbation_rate or default_perturbation_rate
        self.square_scope = square_scope or default_square_scope

    def generate(self, debug=False):

        layer_number = self._generate_layer_number()

        depth = self._generate_depth(layer_number)

        res = self._generate_res(layer_number)

        square = self._generate_square()

        time = self._generate_time()

        _, db_data = loop_tem1d(time=time, L_square=square, depth=depth, res=res, verb_flag=0)
        db_data = np.abs(db_data)  # 加绝对值

        # 增加扰动
        res_data = self.add_perturbation(db_data, self.perturbation_rate)

        if debug is True:
            print("=====depth=====\n", depth)
            print("=====res=====\n", res)
            print("=====square=====\n", square)
            print("=====time=====\n", time)
            print("=====db_data=====\n", db_data)
            print("=====res_data====\n", res_data)

        return {
            'origin_data': db_data,
            'result_data': res_data,
            'time': time,
            'depths': depth,
            'layer_number': layer_number,
            'res': res,
            'square': square,
        }

    @staticmethod
    def add_perturbation(db_data, perturbation_rate):
        # 增加扰动
        # B 数据 在其原始数据的 (-15 ~ 80) % 之间抖动
        # 扰动因子 k = 历史扰动数据的平方和开方分之一
        res_data = [0 for _ in range(len(db_data))]
        k = 1
        # 方案1： 在原始时间上增加扰动
        # for index, value in enumerate(db_data):
        #     perturbation_rate = np.random.randint(perturbation_scope[0], perturbation_scope[1]) \
        #             / 100 / (k ** 0.5)
        #     res_data[index] = value * (1 + perturbation_rate)
        #     k = k + perturbation_rate ** 2
        #

        # 方案2： 在上一个时间增加扰动
        for index, value in enumerate(db_data):
            last_value = db_data[index-1] if index > 0 else db_data[index]
            pr = np.random.randint(
                    -perturbation_rate / 3, perturbation_rate
            ) / 100

            res_data[index] = last_value * (1 + pr)

        return res_data

    def _generate_layer_number(self):
        # print(self.layer_number_scope)
        layer_number_scope = self.layer_number_scope
        layer_number = np.random.randint(layer_number_scope[0], layer_number_scope[1] + 1)

        return layer_number

    def _generate_res(self, layer_number):
        scope = self.resistant_scope

        # 在 0 - 1 内 生成随机数
        pre_res = np.random.random_sample(layer_number + 1)
        # min-max 划入 scope 范围内
        res = scope[0] + pre_res * (scope[1] - scope[0])

        return res

    def _generate_depth(self, layer_number):
        scope = self.layer_number_scope

        pre_depth = np.random.random_sample(layer_number)

        # min-max 划入 scope 范围内
        depths = scope[0] + pre_depth * (scope[1] - scope[0])

        # 生成递增序列
        for i, depth in enumerate(depths):
            if i == 0:
                continue
            depths[i] = depth + depths[i - 1]

        return depths

    def _generate_square(self):
        scope = self.square_scope

        square = np.random.randint(
            scope[0],
            scope[1],
        )

        return square

    def _generate_time(self):
        scope = self.time_sequence_number
        generate_number = np.random.randint(scope[0], scope[1])
        time_range = self.time_scope

        # 方案1: 等距取样
        # times = np.linspace(start=default_time_range[0], stop=default_time_range[1], num=time_sequence_number)

        # 方案2: 对数间隔取样
        min_log_time, max_log_time = log(time_range[0]), log(time_range[1])

        times = np.linspace(start=min_log_time, stop=max_log_time, num=generate_number)
        times = np.exp(times)

        return times


def generate_data(save_dir, file_name, generate_number, generator=None, debug=False):

    if not is_dir_exist(save_dir):
        make_dir_with_input(save_dir)

    generator = generator or Generator()

    checker = Checker()

    for generate_id in range(generate_number):
        logger.info('generating data...')

        data = generator.generate()
        # print(data)
        logger.info('generate succeed!')

        data = EleData(data)        # 模块化

        if debug:
            data.draw()
            sleep(1)

        logger.info('checking teacher')

        if checker.check(data.time, data.origin_data):     # 若通过检查, 则保存

            logger.info('check succeed!')
            logger.info('saving data to file: {}...'.format(file_name))

            data.add_to_dat(save_dir, file_name)

            logger.info('save succeed!')

        else:
            logger.warning('check fail')


def program_run(args):
    """
    Namespace(dma=5.0, dmi=300.0, lma='2', lmi='50', nma=50.0, nmi=2.0, pr='300', rma=10.0, rmi=5.0, tma=5.0, tmi=250.0, **{'generate directory': '/Users/sumbrella/Documents/GitHub/ele_project/gui/data_generate/data'})
    """
    START_FILE_ID = 0
    ONE_FILE_NUMBER = 100
    generator_dir = args.gd
    generator_times = args.gn
    generator = Generator(
        layer_number_scope=[args.lmi, args.lma],
        depth_scope=[args.dmi, args.dma],
        resistant_scope=[args.rmi, args.rma],
        time_sequence_number=[args.nmi, args.nma],
        time_scope=[args.tmi, args.tma],
        square_scope=[args.smi, args.sma]
    )

    for i in range(START_FILE_ID, START_FILE_ID + generator_times):
        generate_data(
            generate_number=ONE_FILE_NUMBER,
            save_dir=generator_dir,
            file_name='LINE_{:03d}_dbdt'.format(i),
            generator=generator,
            debug=args.de
        )



@Gooey(
    program_name='electronic data generator',
)
def main():

    parser = GooeyParser()

    # ================
    # generator directory
    parser.add_argument(
        "gd",
        metavar="generate directory",
        help="Path to the directory you want to generate data",
        widget="DirChooser",
        # metavar="METAVAR"
    )

    # Debug
    parser.add_argument(
        "-de",
        choices=["True", "False"],
        metavar="debug",
        default='False',
    )

    # ================
    # perturbation rate
    parser.add_argument(
        "pr",
        metavar="perturbation rate",
        type=int,
        gooey_options={
            'validator': {
               'test'   : '0 <= int(user_input) <= 80',
               'message': 'Must be between 0 and 80'
            }
        },
        default=50,
    )

    parser.add_argument(
        "gn",
        metavar="generate times",
        default=50,
        help="Times to generate",
        type=int,
    )

    # ================
    # layer number
    layer_scope_group = parser.add_argument_group(
        "layer Number",
        "Customize the maximum and minimum number of layer"
    )

    layer_scope_group.add_argument(
        'lmi',
        metavar='layer minimum',
        # action='count',
        # choices=[1, 2, 3, 4],
        default=2,
        type=int,
        gooey_options={
            'validator': {
                'test'   : '1 <= int(user_input) <= 3',
                'message': 'Must be between 1 and 3'
            }
        }
    )

    layer_scope_group.add_argument(
        'lma',
        metavar='layer maximum',
        # action='count',
        # choices=[3, 4, 5, 6],
        default=5,
        type=int,
        gooey_options={
            'validator': {
                'test'   : '3 <= int(user_input) <= 5',
                'message': 'Must be between 3 and 5'
            }
        }
    )
    # ================
    # Square

    square_group = parser.add_argument_group(
        "Square Scope",
        "Customize the maximum and minimum number of resistant"
    )
    square_group.add_argument(
        'smi',
        metavar='square minimum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '1 <= int(user_input) <= 2',
                'message': 'Must be between 2 and 5'
            }
        },
        default=1,
    )

    square_group.add_argument(
        'sma',
        metavar='square maximum',

        type=int,
        gooey_options={
            'validator': {
                'test'   : '2 <= int(user_input) <= 5',
                'message': 'Must be between 2 and 5'
            },
        },
        default=2,
    )

    # ================
    # Resistant

    resistant_group = parser.add_argument_group(
        "Resistant Scope",
        "Customize the maximum and minimum number of resistant"
    )
    resistant_group.add_argument(
        'rmi',
        metavar='resistant minimum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '10 <= int(user_input) <= 300',
                'message': 'Must be between 10 and 300'
            }
        },
        default=10,
    )

    resistant_group.add_argument(
        'rma',
        metavar='resistant maximum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '200 <= int(user_input) <= 700',
                'message': 'Must be between 200 and 700'
            },
        },
        default=300,
    )

    # ================
    # depth
    depth_group = parser.add_argument_group(
        "Depth Scope",
        "Customize the maximum and minimum number of resistant"
    )

    depth_group.add_argument(
        'dmi',
        metavar='depth minimum',

        type=int,
        gooey_options={
            'validator': {
                'test'   : '5 <= int(user_input) <= 50',
                'message': 'Must be between 5 and 300'
            }
        },
        default=5,
    )

    depth_group.add_argument(
        'dma',
        metavar='depth maximum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '50 <= int(user_input) <= 300',
                'message': 'Must be between 5 and 300'
            }
        },
        default=250,
    )

    # ================
    # time scope
    time_scope_group = parser.add_argument_group(
        "Time Scope",
        "Customize the maximum and minimum number of time"
    )

    time_scope_group.add_argument(
        'tmi',
        metavar='time minimum',
        type=float,
        gooey_options={
            'validator': {
                'test'   : '4 <= float(user_input) <= 6',
                'message': 'Must be between 4 and 6'
            }
        },
        default=4,
    )

    time_scope_group.add_argument(
        'tma',
        metavar='time maximum',
        type=float,
        gooey_options={
            'validator': {
                'test'   : '0 <= float(user_input) <= 3',
                'message': 'Must be between 0 and 3'
            }
        },
        default=2,
    )

    # ================
    # time number
    time_number_group = parser.add_argument_group(
        "Time Number",
        "Customize the maximum and minimum number of time"
    )

    time_number_group.add_argument(
        'nmi',
        metavar='time minimum',

        type=int,
        gooey_options={
            'validator': {
                'test'   : '50 <= int(user_input) <= 150',
                'message': 'Must be between 50 and 150'
            }
        },
        default=50,
    )

    time_number_group.add_argument(
        'nma',
        metavar='time maximum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '150 <= int(user_input) <= 300',
                'message': 'Must be between 150 and 300'
            }
        },
        default=300,
        help="10e5"
    )

    args = parser.parse_args()
    args.tmi = 10 ** (-args.tmi)
    args.tma = 10 ** (-args.tma)
    args.de = True if args.de == 'True' else False
    print(args)

    program_run(args)


if __name__ == '__main__':
    main()
