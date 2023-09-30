FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG CUDA_ARCHS
ARG UID
ARG GID

RUN apt update -y && apt install sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir /workspace
WORKDIR /workspace
RUN groupadd -g $GID user && useradd -u $UID -g $GID user -G sudo -d /workspace
RUN chown user:user /workspace
USER user

RUN sudo apt install -y python3-dev python3-pip
RUN sudo apt install -y libsparsehash-dev git
RUN sudo apt install -y libopenblas-dev
RUN sudo apt install -y cmake
RUN sudo apt install -y cuda-nsight-compute-11-8
RUN sudo apt install -y unzip p7zip-full wget
RUN sudo apt-get clean
RUN sudo rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.0
RUN pip3 install pybind11==2.11.1 cmake==3.27.0 ninja==1.11.1
RUN pip3 install numpy==1.25.2 tqdm==4.65.0 packaging==23.1 pandas==2.0.3 matplotlib==3.7.3 scipy==1.11.1

RUN git clone https://github.com/mit-han-lab/torchsparse.git
COPY assets/torchsparse.patch torchsparse/torchsparse.patch
RUN cd torchsparse  \
    && git checkout 1a10fda15098f3bf4fa2d01f8bee53e85762abcf  \
    && git apply torchsparse.patch && rm torchsparse.patch  \
    && TORCHSPARSE_CUDA_ARCH_LIST="$CUDA_ARCHS" TORCHSPARSE_ENABLE_CUDA=1 CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip install -e .

RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
RUN cd MinkowskiEngine && \
    TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" pip3 install --global-option="--blas=openblas" --global-option="--force_cuda" --global-option="build_ext" --global-option="-j$(nproc)" .

COPY --chown=user source /workspace/Minuet

RUN MINUET_ENABLE_CUDA=1 MINUET_CUDA_ARCH_LIST="$CUDA_ARCHS" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) pip3 install -e /workspace/Minuet

RUN mkdir -p /workspace/artifacts
ENV PYTHONPATH=.
WORKDIR /workspace/artifacts
