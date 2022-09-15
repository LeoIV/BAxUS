FROM python:3.10-buster
WORKDIR /app
COPY . .
RUN apt-get update && \
    apt-get -y upgrade &&\
    apt-get -y install libsuitesparse-dev libatlas-base-dev swig libopenblas-dev libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0 libsdl2-ttf-2.0-0 libsdl2-dev &&\
    pip install --no-cache-dir -r requirements.txt

