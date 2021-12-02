#!/bin/sh
CURRENT=$(pwd)

# Check CUDA_VERSION
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
export TORCH_CUDA_ARCH_LIST="5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6+PTX"

apt update -y && \
    apt install python3 python3-pip git -y && \
    rm -rf /var/lib/apt/lists/*

ln -s /usr/bin/python3 /usr/bin/python

apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    curl \
    unzip \
    wget \
    mc \
    tmux \
    nano \
    build-essential \
    rsync \
    python3-opencv

cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo /work/src > work-directory.pth

cd /work

pip3 install --ignore-installed --no-cache-dir -r requirements.txt

cd /work/src

curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip