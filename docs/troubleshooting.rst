
Troubleshooting
---------------

Mopta08 Executables
^^^^^^^^^^^^^^^^^^^

The executables for the :class:`baxus.benchmarks.real_world_benchmarks.MoptaSoftConstraints`
benchmark are not contained in the repository but downloaded when necessary.
We support four different architectures/operating systems: ARM (32 bit), Windows (64 bit), Linux (64 bit), Linux (32 bit).

The files are automatically downloaded and made executable.
However, this might cause problems if there are no sufficient permissions for writing or making the file executable.
In that case, please download the correct file and move it to ``baxus/benchmarks/mopta08/``.
The files can be downloaded at

* 64bit Windows: `<http://mopta-executables.s3-website.eu-north-1.amazonaws.com/mopta08_amd64.exe>`_
* 32bit ARM: `<http://mopta-executables.s3-website.eu-north-1.amazonaws.com/mopta08_armhf.bin>`_
* 32bit Linux: `<http://mopta-executables.s3-website.eu-north-1.amazonaws.com/mopta08_elf32.bin>`_
* 64bit Linux: `<http://mopta-executables.s3-website.eu-north-1.amazonaws.com/mopta08_elf64.bin>`_

Slice Locatization Data
^^^^^^^^^^^^^^^^^^^

Similarly, it can happen that there are no sufficient permissions to download the slice localization data
for the :class:`baxus.benchmarks.real_world_benchmarks.SVMBenchmark` benchmark.
Please download it from `<http://mopta-executables.s3-website.eu-north-1.amazonaws.com/slice_localization_data.csv.xz>`_
and move it to ``baxus/data/``.