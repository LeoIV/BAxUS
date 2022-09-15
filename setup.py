from setuptools import setup, find_packages

setup(
    name="baxus",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.4",
        "torch>=1.3",
        "LassoBench @ git+https://github.com/ksehic/LassoBench.git",
        "botorch>=0.6",
        "gpytorch<=1.8.1",
        "scikit-learn>=1.1",
        "parameterized>=0.8",
    ],
    exclude_package_data={'': ["results/*", "tests/*"]}
)
