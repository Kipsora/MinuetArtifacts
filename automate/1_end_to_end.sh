GPU=$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv | tail -n 1 | sed -E "s/^NVIDIA GeForce //g" | sed -E 's/ //g' | awk '{print tolower($0)}')

echo "GPU code: $GPU"

function run_e2e {
  dataset=$1
  dataset_label=$2
  batch_size=$3
  model=$4

  python3 scripts/benchmark_end_to_end.py \
    -M "$model" \
    -L minuet \
    --dataset $dataset \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-$model.minuet.json"

  python3 scripts/benchmark_end_to_end.py \
    -M "$model" \
    -L torchsparse \
    --dataset "$dataset" \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-$model.torchsparse.json"

  python3 scripts/benchmark_end_to_end.py \
    -M "$model" \
    -L minkowski \
    --dataset $dataset \
    --batch_size $batch_size \
    --cache_path "./results/$GPU/cache" \
    --json \
    --output "./results/$GPU/$dataset_label-bs=$batch_size-$model.minkowski.json"
}

if [ -d "data/semantic_kitti" ]; then
  run_e2e "configs/semantic_kitti.json" "semantic_kitti" 1 "SparseResNet21D"
  run_e2e "configs/semantic_kitti.json" "semantic_kitti" 1 "SparseResUNet42"
else
  echo "SemanticKITTI dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/semantic3d" ]; then
  run_e2e "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 "SparseResNet21D"
  run_e2e "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 "SparseResUNet42"
else
  echo "Semantic3D dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/s3dis" ]; then
  run_e2e "configs/s3dis.json" "s3dis" 1 "SparseResNet21D"
  run_e2e "configs/s3dis.json" "s3dis" 1 "SparseResUNet42"
else
  echo "Stanford 3D Indoor Scene Dataset (S3DIS) dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

if [ -d "data/shapenet" ]; then
  run_e2e "configs/shapenetsem.json" "shapenetsem" 1 "SparseResNet21D"
  run_e2e "configs/shapenetsem.json" "shapenetsem" 1 "SparseResUNet42"
else
  echo "ShapeNetSem dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

python3 scripts/plot_end_to_end.py
