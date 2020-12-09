import sys
import os


def is_dir_exist(path):
    if not os.path.exists(path):
        print()
        return False
    return True


def make_dir(path):
    input("Create path /")