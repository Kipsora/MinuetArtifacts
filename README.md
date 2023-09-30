# EuroSys 2024 Artifact Evaluation

In this repository, we provide the artifact for the paper 
**Minuet: Accelerating 3D Sparse Convolution on GPUs**.

It is expected to take about **2-3 hours** (excluding 
datasets downloading) to finish all evaluations in the artifact. 

## Hardware & Software Requirements

The artifact should run on any hardware platforms with modern NVIDIA 
desktop/server GPUs (≥ 8 GB GPU memory), x86-64 CPUs, sufficient CPU 
memory (≥ 32 GB), and storage (≥ 150 GB). 
For reference, our experiments are mainly conducted with the following 
hardware specs.

* CPU: AMD Ryzen Threadripper 2920X
* GPU: NVIDIA GeForce RTX 3090 (TDP: 350W)
* Memory: 64 GB DDR4 RAM
* Storage: 256 GB Solid-State Drive (SSD)

The artifact should be executed under a Linux-based operating system with 
up-to-date NVIDIA Driver installed. For reference, we use the following 
software setups:

* OS: Ubuntu 20.04.5 LTS with Linux Kernel 5.15.0-82-generic
* NVIDIA Driver: 535.104.05

Besides, our experiments use 
[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) 
to measure the cache hit ratio,
which requires accesses to GPU performance counters. 
If you experience no permission errors, please follow 
[these instructions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
to grant permissions.

## Step 1. Building the Artifact

We provide the following two options to build this artifact:

### Option 1. Build with Docker Engine (Preferred)

We recommend to use Docker Engine for building the artifact to fully control 
all software dependencies. 
Please follow the instructions to 
[Install Docker Engine](https://docs.docker.com/engine/install/) 
and 
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) 
first.
Note that if the current user is not in the `docker` user group, all following 
docker-related commands requires `root` privilege (i.e. with `sudo`) to run.

To build the docker image, we require to specify the GPU compute capability to 
the docker image, since it is unknown during the building process of the docker 
image. Fortunately, the `nvidia-smi` tool provides a convenient way to check 
the GPU compute capability:

```shell
# Assume we use the first GPU for evaluation, i.e. GPU 0
export GPU_ID=0
nvidia-smi -i $GPU_ID --query-gpu=compute_cap --format=csv
```

The following snippet can build the docker image on the first GPU with `GPU_ID`: 

```shell
# The compute capability can be also set manually. 
# For reference:
#   RTX 2070/2080Ti: export CUDA_ARCHS=7.5
#   RTX 3090: export CUDA_ARCHS=8.6
#   Tesla A100: export CUDA_ARCHS=8.0
export CUDA_ARCHS="$(nvidia-smi -i $GPU_ID --query-gpu=compute_cap --format=csv | tail -n 1)"

# We require the user id and group id to make sure the files we write on the 
# mounted volumes are owned by the current user (i.e. can be cleaned up 
# without sudo). 
docker build \
  --build-arg CUDA_ARCHS=$CUDA_ARCHS \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -t minuet .
```

After successfully built the docker image, execute the following command to 
launch a container for the following experiments/commands:

```shell
docker run -it --rm --gpus $GPU_ID -v "$(pwd):/workspace/artifacts" minuet
```

### Option 2. Build from Source with Anaconda

Please first make sure you have Anaconda installed as described
[here](https://docs.anaconda.com/free/anaconda/install/linux/).

Please execute the following command to create a conda environment for this artifact
and install all software dependencies.

```shell
conda create -n MinuetArtifacts
conda activate MinuetArtifacts
```

Then, please execute the following scripts in under the root of this artifact to 
install all software dependencies and this artifact:

```shell
bash ./automate/conda_install.sh
```

Conda puts the installed dynamic libraries under a different location from the system defaults, 
please use the following command to ensure they will be loaded correctly,

```shell
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

We also require the `PYTHONPATH` environment variable to be set for the scripts to function correctly:

```shell
# We assume every following command will be running under the root of this artifact 
export PYTHONPATH=.
```

## Step 2. Downloading & Preparing the Datasets

We support evaluations on synthetic random datasets and the following real
datasets. Due to each dataset has their own license, we intentionally only
provide instructions to manually download the three datasets as follows:

* (Required, ~253 MB) Semantic3D Dataset
  * Download `birdfountain_station1_xyz_intensity_rgb.7z` to the `data/` folder
    with the following command:
    ```shell
    wget http://www.semantic3d.net/data/point-clouds/testing1/birdfountain_station1_xyz_intensity_rgb.7z -P data 
    ```
* (Optional, ~80 GB) SemanticKITTI Dataset
  * Download the KITTI Odometry Benchmark Velodyne point clouds on
    [the official website](http://www.semantic-kitti.org/dataset.html#download)
    in the `data/` folder, where the downloaded file name should be
    `data_odometry_velodyne.zip`
    ([direct link](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)).
* (Optional, ~5 GB) Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS)
  * Fill the form to retrieve the download links on
    [the official website](http://buildingparser.stanford.edu/dataset.html#Download) ([direct link](https://goo.gl/forms/4SoGp4KtH1jfRqEj2)).
  * Download `Stanford3dDataset_v1.2_Aligned_Version.zip` and place it under
    the `data/` folder.
* (Optional, ~12 GB) ShapeNetSEM Dataset
  * Fill the form to get the permission for downloading the ShapeNetSEM dataset
    [here](https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive).
  * Download `ShapeNetSem.zip` and place it under the `data/` folder.

Note that it is **NOT** required to download all datasets.
However, the experiments depending on an unavailable dataset 
will be skipped, which could cause variability in Figure 10 and Figure 14 
(see [Step 3](#step-3-running-experiments) for details).

Then, please run the following command to decompress and prepare all downloaded 
datasets:

```shell
bash automate/prepare_datasets.sh
```

## Step 3. Running Experiments

We provide instructions for reproducing main performance numbers with the following experiments.

**TL;DR:** Use the following scripts to reproduce all figures for our main evaluations:

```shell
bash automate/run_all.sh
```

In the mapping step experiment E3 (i.e. `3_mapping_step.sh`), some baselines could produce 
Out-Of-Memory (OOM) errors on GPUs with relatively small memory (e.g. RTX 2070 Super). 
These errors are handled in our plotting scripts by producing a visualization plot with the 
subset of the experiments that did not experience OOM error.

The major claims of Minuet C1-C3 (See the artifact appendix A.4.1) hold across different GPU 
models and datasets. However, when reproducing the experiments, the absolute numbers might have 
small fluctuations due to different execution conditions (e.g. GPU thermal management).

Additionally, as mentioned in [Step 2](#step-2-downloading--preparing-the-datasets), it is not 
necessary to download and prepare optional datasets. If the optional datasets are not used, 
this will cause variability in Figure 10 and Figure 14 since these plotted performance numbers 
are averaged over all downloaded and prepared datasets.

The following steps elaborates the details of each experiment.

### Step 3.1 Verification on output consistency with baselines (~5 min)

The following command are used to generate tests for verifying that Minuet has the same 
outputs (within tolerable error) as other baseline frameworks among all datasets:

```shell
python3 scripts/verify.py \
  -L <library> \           # Either "minkowski" or "torchsparse"
  -D <dataset_config> \    # Path to any json file in the configs/ folder
  -M <model> \             # Either "SparseResNet21D" or "SparseResUNet42" (MinkUNet42)
  -T <number_of_tests> \
  --eps <eps>              # The threshold of testing equality of two floating numbers  
```

For example:
```shell
# Verify Minuet's implementation of the SparseResNet21D model 
# with MinkowskiEngine's implementation 
python3 scripts/verify.py \
  -L minkowski \
  -D configs/semantic_kitti.json \
  -M SparseResNet21D \
  -T 5 \
  --eps 1e-6
```

The scripts should generate similar outputs to the following:

```text
Test 0:
Coordinates are equal? True
Features have less than 1e-06 error? True
```

This step automates the verifications, which makes it convenient to see
Minuet's implementation of sparse convolution has exactly the same semantics 
as prior frameworks.

### Step 3.2 End-to-end Performance Comparisons (E1, C1, Figure 9, ~30 min)

The following scripts collects raw performance numbers for end-to-end evaluations:

```shell
python3 scripts/benchmark_end_to_end.py \
  -D <dataset_config> \    # Path to any json file in the configs/ folder
  -L <library> \           # Either "minkowski", "torchsparse", or "minuet"
  -M <model>               # Either "SparseResNet21D" or "SparseResUNet42" (MinkUNet42)
```

For example:
```shell
python3 scripts/benchmark_end_to_end.py \
  -D configs/semantic_kitti.json \
  -L minuet \
  -M SparseResNet21D
```

The command should generate a table similar to the following:

```text
| Field   | Value              |
+=========+====================+
| latency | 12.548677363557816 |
```

The `latency` field shows the average latency (in milliseconds) of the end-to-end 
inference of the given model.

We provide a script to automatically collect all results needed to reproduce 
the end-to-end performance evaluation figure.

```shell
bash automate/1_end_to_end.sh
```

The generated figure will be at `figures/figure9_end_to_end_speed_up.<gpu>.pdf` where `<gpu>` denotes the GPU 
used for the benchmarks.

### Step 3.3 Layer-wise performance comparisons (E2, C1, Figure 10, ~30 min)

The following scripts collects raw performance numbers for layer-wise evaluations.

```shell
python3 scripts/benchmark_layerwise.py \
  -D <dataset_config> \      # Path to any json file in the configs/ folder
  -L <library> \             # Either "minkowski", "torchsparse", or "minuet"
  -K <kernel_size> \         # The kernel size for the layer
  --channels <in_channels> <out_channels> # number of input and output channels
```

For example,

```shell
python3 scripts/benchmark_layerwise.py \
  -D configs/semantic_kitti.json \
  -L minuet \
  -K 3 \
  --channels 32 32
```

The command above generate a table similar to the following:

```text
| Field        | Value                |
+==============+======================+
| latency_full | 0.8304032027721405   |
| latency_gmas | 0.45921599864959717  |
```

The `latency_full` field shows the average latency (in milliseconds) of the layer
execution with the given channel sizes. The `latency_gmas` field shows the latency of 
the Gather-MatMul-Scatter step, which is used for [Step 3.5](#step-35-gather-matmul-scatter-step-performance-evaluations-e4-c3-figure-14-1-min).

We provide a script to automatically collect all results needed to reproduce
the layer-wise performance evaluation figure.

```shell
bash automate/2_layerwise_gather_matmul_scatter_step.sh
```

The generated figure will be at `figures/figure10_layer_wise_speedup.<gpu>.pdf` 
where `<gpu>` denotes the GPU used for the benchmarks.

### Step 3.4 Mapping step performance comparisons (E3, C2, Figure 11 & 12, ~25 min)

The following scripts collects raw performance numbers for the mapping step evaluations.

```shell
python3 scripts/benchmark_mapping.py \
  -D <dataset_config> \      # Path to any json file in the configs/ folder
  -L <library> \             # Either "minkowski", "torchsparse", or "minuet"
  -K <kernel_size>           # The kernel size for the layer
```

For example,

```shell
python3 scripts/benchmark_mapping.py \
  -D configs/semantic_kitti.json \
  -L minuet \
  -K 3
```

will generate the following table:

```text
| Field         | Value               |
+===============+=====================+
| latency_build | 0.18038080185651778 |
| latency_query | 0.14938880056142806 |
```

The `latency_query` field shows the average latency (in milliseconds) of querying
(for building kernel maps).
The `latency_build` field shows the average latency (in milliseconds) of building
the sorted tables (Minuet) or hash tables (MinkowskiEngine, TorchSparse) for 
the queries.

We provide a script to automatically collect all results needed to reproduce
the mapping step performance evaluation figure.

```shell
bash automate/3_mapping_step.sh
```

Along with the JSON-formatted results generated in folder `results/`, three figures will
be generated according to the results, `figures/figure11a_mapping_hit_ratio.<gpu>.pdf`,
`figures/figure11b_mapping_query_time.<gpu>.pdf`, and `figures/figure12_mapping_build_time.<gpu>.pdf`,
which respectively corresponds to the L2 hit ratio, query and build latencies.

If the Figure 11a is not generated, this is probably due to not having permission to access
the GPU performance counters. Please check if the `ERR_NVGPUCTRPERM` error appears in any 
of the `results/*.csv` files and (if yes) follow the instructions in 
[Hardware & Software Requirements](#hardware--software-requirements) 
to enable access to these performance counters. 

### Step 3.5 Gather-MatMul-Scatter step performance evaluations (E4, C3, Figure 14, ~1 min)

As described in [Step 3.3](#step-33-layer-wise-performance-comparisons-e2-c1-figure-10-30-min), 
the performance numbers for the Gather-Matmul-Scatter step is already collected 
in layer-wise evaluations. 
Thus, in this step no further action is required.

The scripts for reproducing the layer-wise performance figure will also generate the figure 
for the evaluations of the Gather-MatMul-Scatter step, which is located at 
`figures/figure14_gather_gemm_scatter_layerwise_speedup.<gpu>.pdf`
where `<gpu>` denotes the GPU used for the benchmarks.

## License

Please refer to [LICENSE](LICENSE) for details.
