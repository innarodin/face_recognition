FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="rodinskayaia"

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
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

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        Pillow \
        imutils \
        pymemcache \
        h5py \
        matplotlib \
        numpy==1.14.5 \
        pandas \
        scipy \
        mxnet-cu90 \
        sklearn \
	opencv-python \
	opencv-contrib-python \
	pika \
	configobj \
	py-postgresql

RUN pip3 --no-cache-dir install redis \
	redisworks

# Install TensorFlow GPU version.
RUN pip3 --no-cache-dir install \
    tensorflow_gpu==1.10.1

RUN apt-get update && apt-get install -y git

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ENV TZ=Asia/Yekaterinburg

# TensorBoard
EXPOSE 6007

WORKDIR "/app"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
