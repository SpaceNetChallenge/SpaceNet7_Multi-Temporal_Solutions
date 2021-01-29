FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    #libgdal-dev \
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
	tmux \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda update conda && \
    #conda config --remove channels defaults && \
    conda config --add channels conda-forge

# set up conda environment and add to $PATH
RUN conda create -n spacenet7 python=3.8 \
                    && echo "source activate spacenet7" > ~/.bashrc
ENV PATH /opt/conda/envs/spacenet7/bin:$PATH

RUN mkdir -p /root/.torch/models

RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN pip install albumentations
RUN pip install pretrainedmodels
RUN pip install timm

RUN pip install tensorboardX \
	&& pip install torchsummary \
	&& pip install utm \
	&& pip install numba

RUN pip uninstall apex
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  /apex
RUN conda install -n spacenet7 \
			  #gdal=2.4.2 \
		   	  geopandas \
			  #fiona \
			  rasterio \
              #osmnx=0.10 \
     	      awscli \
              affine \
              pyhamcrest \
              cython \
              #fiona \
              h5py \
              ncurses \
              jupyter \
              jupyterlab \
              ipykernel \
              #libgdal \
              matplotlib \
	          ncurses \
              numpy \
			  statsmodels \
              opencv \
              pandas \
              pillow \
              pip \
              scipy \
              scikit-image \
              scikit-learn \
              shapely \
              #gdal \
              rtree \
              testpath \
              tqdm \
              pandas \
              geopandas \
              rasterio \
			  #opencv=4.0.0 \
			  opencv \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all
RUN pip install pyclipper cached_property efficientnet_pytorch geojson

ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate spacenet7 && python -m ipykernel.kernelspec
# speeding up weights downloading from personal dropbox
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnet_b7_ns-1dbc32de.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnet_b6_ns-51548356.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006
WORKDIR /work

# get trained model weights
RUN mkdir weights
RUN mkdir logs

RUN wget -O weights/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b6_ns_0_best_dice https://www.dropbox.com/s/u3ql3czyjug7j2k/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b6_ns_0_best_dice?dl=0
RUN wget -O weights/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice https://www.dropbox.com/s/jbfz0dfdbysjabm/run1_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice?dl=0
RUN wget -O weights/run2_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice https://www.dropbox.com/s/39uu0689ikaz63n/run2_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice?dl=0
RUN wget -O weights/run3_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice https://www.dropbox.com/s/wnk794pfkcyvrqb/run3_mask_center_timm_effnet_dragon_tf_efficientnet_b7_ns_0_best_dice?dl=0

COPY . /work/


RUN chmod 777 train.sh
RUN chmod 777 test.sh

RUN ["/bin/bash"]