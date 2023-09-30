GPU=$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv | tail -n 1 | sed -E "s/^NVIDIA GeForce //g" | sed -E 's/ //g' | awk '{print tolower($0)}')

echo "GPU code: $GPU"

function run_layerwise {
  dataset=$1
  dataset_label=$2
  batch_size=$3
  in_channels=$4
  out_channels=$5
  kernel_size=$6

  python3 scripts/benchmark_layerwise.py \
    -L "torchsparse" \
    --channels $in_channels $out_channels \
    --kernel_size $kernel_size \
    --dataset "$dataset" \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-c=${in_channels}x${out_channels}-ks=$kernel_size.torchsparse.json"

  python3 scripts/benchmark_layerwise.py \
    -L "minuet" \
    --channels $in_channels $out_channels \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-c=${in_channels}x${out_channels}-ks=$kernel_size.minuet.json"

  python3 scripts/benchmark_layerwise.py \
    -L "minkowski" \
    --channels $in_channels $out_channels \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-c=${in_channels}x${out_channels}-ks=$kernel_size.minkowski.json"
}


if [ -d "data/semantic_kitti" ]; then
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 8 8 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 16 16 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 32 32 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 64 64 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 128 128 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 256 256 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 4 16 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 64 96 3
  run_layerwise "configs/semantic_kitti.json" "semantic_kitti" 1 256 384 3
else
  echo "SemanticKITTI dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/semantic3d" ]; then
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 8 8 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 16 16 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 32 32 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 64 64 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 128 128 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 256 256 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 4 16 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 64 96 3
  run_layerwise "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 256 384 3
else
  echo "Semantic3D dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/s3dis" ]; then
  run_layerwise "configs/s3dis.json" "s3dis" 1 8 8 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 16 16 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 32 32 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 64 64 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 128 128 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 256 256 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 4 16 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 64 96 3
  run_layerwise "configs/s3dis.json" "s3dis" 1 256 384 3
else
  echo "Stanford 3D Indoor Scene Dataset (S3DIS) dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/shapenet" ]; then
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 8 8 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 16 16 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 32 32 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 64 64 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 128 128 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 256 256 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 4 16 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 64 96 3
  run_layerwise "configs/shapenetsem.json" "shapenetsem" 1 256 384 3
else
  echo "ShapeNetSem dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

python3 scripts/plot_layerwise.py
python3 scripts/plot_gmas_step.py
