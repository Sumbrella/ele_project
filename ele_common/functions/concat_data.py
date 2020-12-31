import os

from loguru import logger

from ele_common.units import SingleFile
from ele_common.units import SinglePoint
from ele_common.functions import fcheck


def concat_dir(dir_path, full_res_path=None, overwrite=True, extension="dat"):

    if full_res_path is None:
        full_res_path = os.path.join(dir_path, "result.dat")

    if not fcheck.is_path_exist(dir_path):
        raise ValueError("Such {} do not exist".format(dir_path))

    res_dir = os.path.dirname(full_res_path)

    if not fcheck.is_path_exist(res_dir):
        raise ValueError("Such {} do not exist".format(res_dir))

    if overwrite and fcheck.is_path_exist(full_res_path):
        os.remove(full_res_path)

    target_files = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if fcheck.check_ext(file, extension)
    ]

    concat_files(target_files, full_res_path)


def concat_files(files, res_path):
    for file in files:
        logger.info("handling file {}".format(file))
        sf = SingleFile(file)
        for _, point in enumerate(sf.point_reader()):
            if point.size == 0:
                continue
            point.add_to_file(res_path)
            logger.info(f"process {_}/{sf.point_number}")


if __name__ == '__main__':
    # concat_dir(dir_path="../../data/generate/data", full_res_path="../../data/generate/concat/data_result.dat")
    # concat_dir(dir_path="../../data/generate/teacher", full_res_path="../../data/generate/concat/teacher_result.dat")
    pass