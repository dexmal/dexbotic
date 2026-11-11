FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i -e "s/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    sed -i -e "s/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && apt-get update -y && \
    apt-get install --assume-yes --fix-missing build-essential && \
    apt-get install -y openssh-server vim git curl tmux && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


WORKDIR /app

COPY dexbotic/ /app/dexbotic/
COPY pyproject.toml /app/pyproject.toml

RUN /opt/conda/bin/conda create -n dexbotic python=3.10 -y && \
    /bin/bash -c "source activate dexbotic && \
        pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu118 && \
        pip install -e . && pip install transformers==4.57.6"

RUN /bin/bash -c "source activate dexbotic && \
        pip install ninja && \
        pip install packaging && \
        MAX_JOBS=2 pip install flash-attn --no-build-isolation"

RUN /opt/conda/bin/conda init bash

CMD ["bash"]