FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="rodinskayaia"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
	libsm6 \
	libxrender1 \
	libfontconfig1 \
        pkg-config \
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
	postgresql \
	postgresql-contrib \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TZ=Asia/Yekaterinburg

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        imutils \
        h5py \
        numpy==1.14.5 \
        scipy \
	pika \
	configobj \
	py-postgresql \
	redis \
	redisworks

RUN apt-get update && apt-get install -y git

# TensorBoard
EXPOSE 6008

WORKDIR "/app"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
