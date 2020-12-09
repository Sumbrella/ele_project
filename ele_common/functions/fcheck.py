import os
from loguru import logger


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

