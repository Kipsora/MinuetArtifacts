GPU=$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv | tail -n 1 | sed -E "s/^NVIDIA GeForce //g" | sed -E 's/ //g' | awk '{print tolower($0)}')

echo "GPU code: $GPU"

function run_mapping {
  dataset=$1
  dataset_label=$2
  batch_size=$3
  kernel_size=$4

  python3 scripts/benchmark_mapping.py \
    -L "torchsparse" \
    --kernel_size $kernel_size \
    --dataset "$dataset" \
    --num_rounds 100 \
    --batch_size $batch_size \
    --json \
    --output "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.torchsparse.json"

  python3 scripts/benchmark_mapping.py \
    -L "minuet" \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --num_rounds 100 \
    --batch_size $batch_size \
    --json \
    --output "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.minuet.json"

  python3 scripts/benchmark_mapping.py \
    -L "minkowski" \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --num_rounds 100 \
    --batch_size $batch_size \
    --json \
    --output "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.minkowski.json"
}

function collect_ncu() {
  report=$1
  shift
  launch_skip=$1
  shift
  kernel_name=$1
  shift
  ncu \
    --csv \
    --force-overwrite \
    --target-processes all \
    --replay-mode kernel \
    --kernel-name $kernel_name \
    --kernel-name-base function \
    --launch-skip $launch_skip \
    --launch-skip-before-match 0 \
    --launch-count 1 \
    --kill no \
    --filter-mode global \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section LaunchStats \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --section Nvlink_Tables \
    --section Nvlink_Topology \
    --section Occupancy \
    --section SchedulerStats \
    --section SourceCounters \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section WarpStateStats \
    --sampling-interval auto \
    --sampling-max-passes 5 \
    --sampling-buffer-size 33554432 \
    --profile-from-start 1 \
    --cache-control all \
    --clock-control base \
    --apply-rules yes \
    --import-source no \
    --check-exit-code yes \
    "$@" > "$report"
}

function run_mapping_with_ncu() {
  dataset=$1
  dataset_label=$2
  batch_size=$3
  kernel_size=$4

  mkdir -p "./results/$GPU"

  collect_ncu "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.minuet.csv" 3 QuerySortedIndexWithOffsets \
    python3 scripts/benchmark_mapping.py \
    -L "minuet" \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --num_rounds 2 \
    --batch_size $batch_size \
    --verbose

  collect_ncu "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.torchsparse.csv" 3 cuckooLookupKernel_Multi \
    python3 scripts/benchmark_mapping.py \
    -L "torchsparse" \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --num_rounds 2 \
    --batch_size $batch_size \
    --verbose

  collect_ncu "./results/$GPU/mapping-$dataset_label-bs=$batch_size-ks=$kernel_size.minkowski.csv" 3 direct_kernel_map \
    python3 scripts/benchmark_mapping.py \
    -L "minkowski" \
    --kernel_size $kernel_size \
    --dataset $dataset \
    --num_rounds 2 \
    --batch_size $batch_size \
    --verbose
}

if [ -d "data/semantic3d" ]; then
  run_mapping "configs/semantic3d-birdfountain-0.01.json" "semantic3d-birdfountain-0.01" 1 3
  run_mapping "configs/semantic3d-birdfountain-0.02.json" "semantic3d-birdfountain-0.02" 1 3
  run_mapping "configs/semantic3d-birdfountain-0.05.json" "semantic3d-birdfountain-0.05" 1 3
  run_mapping "configs/semantic3d-birdfountain-0.08.json" "semantic3d-birdfountain-0.08" 1 3
  run_mapping "configs/semantic3d-birdfountain-0.1.json" "semantic3d-birdfountain-0.1" 1 3
  run_mapping "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 3

  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.01.json" "semantic3d-birdfountain-0.01" 1 3
  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.02.json" "semantic3d-birdfountain-0.02" 1 3
  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.05.json" "semantic3d-birdfountain-0.05" 1 3
  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.08.json" "semantic3d-birdfountain-0.08" 1 3
  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.1.json" "semantic3d-birdfountain-0.1" 1 3
  run_mapping_with_ncu "configs/semantic3d-birdfountain-0.2.json" "semantic3d-birdfountain-0.2" 1 3
else
  echo "Semantic3D dataset is not installed."
  echo "Experiments related to this dataset will be skipped"
fi

run_mapping "configs/random-5e4-f1-s=0.2.json" "random-5e4-f1-s=0.2" 1 3
run_mapping "configs/random-1e5-f1-s=0.2.json" "random-1e5-f1-s=0.2" 1 3
run_mapping "configs/random-5e5-f1-s=0.2.json" "random-5e5-f1-s=0.2" 1 3
run_mapping "configs/random-1e6-f1-s=0.2.json" "random-1e6-f1-s=0.2" 1 3
run_mapping "configs/random-5e6-f1-s=0.2.json" "random-5e6-f1-s=0.2" 1 3
run_mapping "configs/random-1e7-f1-s=0.2.json" "random-1e7-f1-s=0.2" 1 3

run_mapping_with_ncu "configs/random-5e4-f1-s=0.2.json" "random-5e4-f1-s=0.2" 1 3
run_mapping_with_ncu "configs/random-1e5-f1-s=0.2.json" "random-1e5-f1-s=0.2" 1 3
run_mapping_with_ncu "configs/random-5e5-f1-s=0.2.json" "random-5e5-f1-s=0.2" 1 3
run_mapping_with_ncu "configs/random-1e6-f1-s=0.2.json" "random-1e6-f1-s=0.2" 1 3
run_mapping_with_ncu "configs/random-5e6-f1-s=0.2.json" "random-5e6-f1-s=0.2" 1 3
run_mapping_with_ncu "configs/random-1e7-f1-s=0.2.json" "random-1e7-f1-s=0.2" 1 3

python3 scripts/plot_map_step_query.py
python3 scripts/plot_map_step_build.py
python3 scripts/plot_map_step_l2.py
