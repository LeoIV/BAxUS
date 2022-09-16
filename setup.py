from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name="BAxUS",
    version="0.0.8",
    author="Leonard Papenmeier",
    author_email="leonard.papenmeier@cs.lth.se",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.4",
        "torch>=1.3",
        "lasso-bench-fork-leoiv==0.0.6",
        "botorch>=0.6",
        "gpytorch<=1.8.1",
        "scikit-learn>=1.1",
        "parameterized>=0.8",
    ],
    exclude_package_data={'': ["results/*", "tests/*"]},
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': ['benchmark-runner=baxus.util.console_entry_point:bench'],
    }
)
