from loguru import logger

from ele_common.units import Generator
from ele_common.units import EleData
from ele_common.units import Checker
from ele_common.functions import is_dir_exist
from ele_common.functions import make_dir_with_input


def generate_data(save_dir, file_name, generate_number):

    if not is_dir_exist(save_dir):
        make_dir_with_input(save_dir)

    generator = Generator()
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


if __name__ == '__main__':
    START_FILE_ID = 1
    ONE_FILE_NUMBER = 100
    for id in range(START_FILE_ID, 3000):
        generate_data(
            generate_number=ONE_FILE_NUMBER,
            save_dir='../../data/generate',
            file_name='LINE_{:04d}_dbdt'.format(id)
        )
