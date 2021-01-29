import os.path
import sys


def init_path():
    """[summary]
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.abspath(os.path.join(this_dir, '..'))
    sys.path.append(proj_dir)


init_path()
