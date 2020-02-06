FROM ubuntu:18.04

RUN apt -y update && \
    apt -y upgrade && \
    apt -qq -y install libsm6 libxext6 libxrender-dev python3 python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-cpu==2.1.0 opencv-python matplotlib seaborn numpy pandas

