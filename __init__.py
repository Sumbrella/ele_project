from __future__ import absolute_import, division, print_function
import sys

hard_dependencies = ('matplotlib', 'csv', 'loguru', 'numpy', 'empymod', 'scipy', 'sklearn', 'argparse')
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies


sys.path.append(".")
