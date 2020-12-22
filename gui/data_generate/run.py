from loguru import logger
from gooey import Gooey, GooeyParser

from ele_common.units import Generator
from ele_common.units import EleData
from ele_common.units import Checker
from ele_common.functions import is_dir_exist
from ele_common.functions import make_dir_with_input


def generate_data(save_dir, file_name, generate_number, generator=None):

    if not is_dir_exist(save_dir):
        make_dir_with_input(save_dir)

    generator = generator or Generator()
    checker = Checker()

    for generate_id in range(generate_number):
        logger.info('generating data...')

        data = generator.generate()

        logger.info('generate succeed!')

        data = EleData(data)        # 模块化

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
    print([args.lmi, args.lma])
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
        )



@Gooey(
    advanced=True,
    auto_start=True,
    program_name='electronic data generator',
)
def main():

    parser = GooeyParser()

    parser.add_argument(
        "gd",
        metavar="generate directory",
        help="Path to the directory you want to generate data",
        widget="DirChooser",
        # metavar="METAVAR"
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
                'message': 'Must be between 10 and 700'
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
        metavar='resistant minimum',

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
        metavar='resistant maximum',
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

        type=int,
        gooey_options={
            'validator': {
                'test'   : '4 <= int(user_input) <= 5',
                'message': 'Must be between 4 and 5'
            }
        },
        default=5,
    )

    time_scope_group.add_argument(
        'tma',
        metavar='time maximum',
        type=int,
        gooey_options={
            'validator': {
                'test'   : '2 <= int(user_input) <= 3',
                'message': 'Must be between 2 and 3'
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

    args = parser.parse_args()

    print(args)

    program_run(args)


if __name__ == '__main__':
    main()