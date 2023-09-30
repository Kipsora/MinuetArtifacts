set -eu

conda install -y python=3.9 cmake ninja openblas-devel wget -c anaconda
conda install -y cuda-toolkit cuda-nvcc cuda-tools -c "nvidia/label/cuda-11.8.0"
conda install -y gcc=11 gcc_linux-64=11 gxx=11 gxx_linux-64=11 sysroot_linux-64=2.17 binutils sparsehash p7zip unzip -c conda-forge

# Fix the nsight compute problem in conda installation
ln -s "$CONDA_PREFIX/nsight-compute/2022.3.0" "$CONDA_PREFIX/nsight-compute-2022.3.0"

yes | pip3 install torch --index-url https://download.pytorch.org/whl/cu118
yes | pip3 install pybind11==2.11.1 cmake==3.27.0 ninja==1.11.1 numpy==1.25.2 tqdm==4.65.0 packaging==23.1 pandas==2.0.3 matplotlib==3.7.3 scipy==1.11.1

export CUDA_HOME=$CONDA_PREFIX

git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++) \
  pip3 install --global-option="--blas=openblas" --global-option="--force_cuda" --global-option="build_ext" --global-option="-j$(nproc)" .
cd ..

git clone https://github.com/mit-han-lab/torchsparse
cd torchsparse
git checkout 1a10fda15098f3bf4fa2d01f8bee53e85762abcf
git apply ../assets/torchsparse.patch
CC=$(which gcc) \
CXX=$(which g++) \
CUDAHOSTCXX=$(which g++) \
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  pip3 install -e .
cd ..

# Install Minuet
cd source
CC=$(which gcc) \
CXX=$(which g++) \
CUDAHOSTCXX=$(which g++) \
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  pip3 install -e .
cd ..
