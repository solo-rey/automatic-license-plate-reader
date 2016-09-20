FROM ubuntu:14.04

RUN apt-get update && \
   apt-get install -y \
   build-essential \
   cmake \
   git \
   wget \
   unzip \
   pkg-config \
   libswscale-dev \
   python3-dev \
   python3-numpy \
   python3-matplotlib \
   python3-pip \
   libtbb2 \
   libtbb-dev \
   libjpeg-dev \
   libpng-dev \
   libtiff-dev \
   libjasper-dev \
   libavformat-dev \
   tesseract-ocr \
   python3-pillow \
   && apt-get -y clean all \
   && rm -rf /var/lib/apt/lists/*

RUN pip3 install jupyter pytesseract

RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip \
   && unzip 3.1.0.zip \
   && mkdir /opencv-3.1.0/cmake_binary \
   && cd /opencv-3.1.0/cmake_binary \
   && cmake .. \
   && make install \
   && rm /3.1.0.zip \
   && rm -r /opencv-3.1.0
