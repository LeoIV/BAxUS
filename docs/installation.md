# Installation
We explain how to install BAxUS and how to build the docs.

## Installation

You have 3 options for installing `BAxUS`.
Please make sure to install the following packages before running the `BAxUS` installation.
We assume that you have a Debian Buster based Linux distribution. Please use a Docker image 
if you are working with a different distribution:
```
apt-get update && apt-get -y upgrade && apt-get -y install libsuitesparse-dev libatlas-base-dev swig libopenblas-dev libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0 libsdl2-ttf-2.0-0 libsdl2-dev
```

### `pip` installation

``
python3 -m pip install baxus
``

### Installation from source

First install required software:

```bash
apt-get update && apt-get -y upgrade && apt-get -y install libsuitesparse-dev libatlas-base-dev swig libopenblas-dev libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0 libsdl2-ttf-2.0-0 libsdl2-dev
```

Then install with the `setup.py`:

```bash
cd baxus
pip install .
```

or with the requirements.txt:

```bash
cd baxus
pip install -r requirements.txt
```

### Docker image

Alternatively, use the Docker installation.
We do not share the Docker image to ensure anonymity.
However, you can build the Docker image yourself with the provided `Dockerfile`:

First, [install Docker](https://docs.docker.com/engine/install/).
Next, build the Docker image

```bash
cd baxus
sudo docker build -t baxus
```

By default, BAxUS stores all results in a directory called `results`.
To get the results on the host machine, first create this directory and mount it into the Docker container:

```bash
mkdir results
sudo docker run -v "$(pwd)/results":/app/results baxus /bin/bash -c "python benchmark_runner.py -id 100 -td 1 -f branin2 --adjust-initial-target-dimension"
```

After the run completed, the results can be obtained in the `./results` directory.

## Building the docs

To build the docs, you need to install additional packages:
```bash
pip install sphinx m2r2 sphinx_rtd_theme
```
If you want to build the PDF documentation, you further need to install
```bash
sudo apt-get install texlive texlive-latex-extra latexmk
```

The docs are located in the docs directory. 
To build the API doc, run
```bash
cd docs
sphinx-apidoc -o . ../baxus
```

To build the HTML version, run
```bash
make html
```
and for the PDF version,
```bash
make latexpdf
```

The docs are located in `docs/_build/html` or `docs/_build/pdf`.