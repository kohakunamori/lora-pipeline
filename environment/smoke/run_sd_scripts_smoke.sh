#!/usr/bin/env bash
set -Eeuo pipefail

batch="${1:-1}"
case "$batch" in
  1|2) ;;
  *) echo "usage: $0 1|2" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${LORA_PIPELINE_ROOT:-$(cd "$script_dir/../.." && pwd)}"
python="${LORA_PYTHON:-$root/.conda/envs/lora-pipeline/bin/python}"
accelerate="${LORA_ACCELERATE:-$(dirname "$python")/accelerate}"
sd_scripts="${LORA_SD_SCRIPTS:-$root/vendor/sd-scripts}"
dataset_config="${LORA_SMOKE_DATASET_CONFIG:-$script_dir/dataset-batch${batch}.toml}"
train_config="${LORA_SMOKE_TRAIN_CONFIG:-$script_dir/train.toml}"
timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${LORA_SMOKE_RUN_ROOT:-$root/smoke/runs}/batch${batch}-${timestamp}"
output_dir="$run_dir/output"
log_dir="$run_dir/logs"
monitor_pid=""

for executable in "$python" "$accelerate"; do
  if [[ ! -x "$executable" ]]; then
    echo "required executable is missing: $executable" >&2
    exit 2
  fi
done
for file in "$sd_scripts/sdxl_train_network.py" "$dataset_config" "$train_config"; do
  if [[ ! -f "$file" ]]; then
    echo "required file is missing: $file" >&2
    exit 2
  fi
done

cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

mkdir -p "$output_dir" "$log_dir"
echo "This smoke test assumes exclusive access to the selected GPU." >&2

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop-ms=200 >"$log_dir/gpu-monitor.csv" 2>&1 &
monitor_pid=$!

export HF_HOME="${HF_HOME:-$root/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONUNBUFFERED=1

start_epoch="$(date +%s)"
set +e
(
  cd "$sd_scripts"
  "$accelerate" launch \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 1 \
    --mixed_precision fp16 \
    sdxl_train_network.py \
    --config_file "$train_config" \
    --dataset_config "$dataset_config" \
    --output_dir "$output_dir" \
    --output_name "v100-batch${batch}-smoke" \
    --logging_dir "$log_dir/tensorboard"
) 2>&1 | tee "$log_dir/train.log"
train_rc=${PIPESTATUS[0]}
set -e
end_epoch="$(date +%s)"

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""

peak_mib="$(awk -F, '
  { value=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); if (value+0 > peak) peak=value+0 }
  END { print peak+0 }
' "$log_dir/gpu-monitor.csv")"
artifact_count="$(find "$output_dir" -maxdepth 1 -type f -name '*.safetensors' | wc -l)"
elapsed_seconds=$((end_epoch - start_epoch))

echo "SMOKE_RESULT batch=$batch rc=$train_rc peak_mib=$peak_mib elapsed_seconds=$elapsed_seconds artifacts=$artifact_count run_dir=$run_dir"

if [[ "$train_rc" -ne 0 || "$artifact_count" -lt 1 ]]; then
  exit 1
fi
