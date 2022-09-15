import sys
from logging import warning

from baxus.benchmark_runner import main

import ctypes

if __name__ == "__main__":
    try:
        libgcc_s = ctypes.CDLL('libgcc_s.so.1')
    except Exception as e:
        warning("Failed to load libgcc_s.so.1 explicitly. This can lead to a weird error where Saasbo cannot "
                "be executed. See https://stackoverflow.com/a/65908383/3736965")
    main(sys.argv[1:])
