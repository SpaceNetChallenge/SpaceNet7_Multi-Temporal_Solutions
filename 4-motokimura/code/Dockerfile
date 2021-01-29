# Adapted from: https://github.com/CosmiQ/CosmiQ_SN7_Baseline/blob/master/docker/Dockerfile

FROM nvidia/cuda:10.2-devel-ubuntu16.04
LABEL maintainer="motokimura <motoki.kimura.1990@gmail.com>"

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ARG solaris_branch='master'

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    sudo \
    libx11-6 \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    vim \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

# prepend pytorch and conda-forge before default channel
RUN conda update conda && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch

# get dev version of solaris and create conda environment based on its env file
WORKDIR /tmp/
RUN git clone https://github.com/motokimura/solaris.sn7 solaris && \
    cd solaris && \
    git checkout ${solaris_branch} && \
    conda env create -f environment-gpu.yml
ENV PATH /opt/conda/envs/solaris/bin:$PATH

RUN cd solaris && pip install .

# install various conda dependencies into the conda environment
RUN conda install -n solaris \
    jupyter \
    jupyterlab \
    ipykernel

# install segmentation_models.pytorch
RUN pip install git+https://github.com/motokimura/segmentation_models.pytorch.sn7  # install latest master
RUN pip install yacs  # required to use spacenet7_model library
RUN pip install gitpython tensorboard tensorboardX lightgbm==2.3.1 geomet==0.2.1-1  # required to run scripts under tools/
RUN pip install seaborn  # required to run notebooks under notebooks/
RUN pip install imagecodecs  # required for Planet files

# XXX: temporally commented out to avoid nodejs error
# to show tqdm progress bar on jupyter lab nicely
#RUN conda install -n solaris -y -c conda-forge nodejs
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate solaris && python -m ipykernel.kernelspec \
    --name solaris --display-name solaris

# activate conda environment immediately after entering container
RUN echo "conda activate solaris" >> ~/.bashrc

# copy files
COPY configs /work/configs
COPY scripts /work/scripts
COPY spacenet7_model /work/spacenet7_model
COPY static /work/static
COPY tools /work/tools
COPY *.sh /work/

# set permissions for execution of shell/python scripts
RUN chmod a+x /work/scripts/test/*.sh
RUN chmod a+x /work/scripts/train/*.sh
RUN chmod a+x /work/tools/*.py
RUN chmod a+x /work/*.sh
ENV PATH $PATH:/work/

# download pretrained models
# these models are removed before training (see train.sh)
WORKDIR /work/models
RUN wget https://motokimura-public-sn7.s3.amazonaws.com/logs.zip
RUN wget https://motokimura-public-sn7.s3.amazonaws.com/weights.zip
RUN unzip logs.zip && rm logs.zip
RUN unzip weights.zip && rm weights.zip

WORKDIR /work