from __future__ import division, absolute_import, print_function

hard_dependencies = ('matplotlib', 'csv', 'loguru', 'numpy', 'empymod', 'scipy', 'sklearn')
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
