#!/usr/bin/env bash
# Single-node paper recipe: 8 MI300X; TP2 × CP4 × DP1, EP8.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
: "${SFT_DATA:?Path to cleaned SFT JSONL}"
: "${STUDENT_HF:?Path to Qwen3.5-35B-A3B-Base HF checkpoint}"
: "${STUDENT_REF:?Path to converted Megatron checkpoint}"
: "${SAVE_DIR:?New training output directory}"
: "${MEGATRON_PATH:?Compatible Megatron-LM source directory}"
for path in "$SFT_DATA" "$STUDENT_HF/config.json" "$STUDENT_REF/latest_checkpointed_iteration.txt" "$MEGATRON_PATH/megatron/training"; do
  test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
source "$REPO_DIR/training/model.sh"
export PYTHONPATH="$REPO_DIR/training/backend:$MEGATRON_PATH:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 NVTE_USE_ROCM=1 HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 CUDA_DEVICE_MAX_CONNECTIONS=1
export SLIME_DIST_TIMEOUT_MINUTES=60
TRAIN_ARGS=(
  --actor-num-nodes 1 --actor-num-gpus-per-node 8
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "$STUDENT_HF" --ref-load "$STUDENT_REF" --save "$SAVE_DIR" --save-interval 20
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "$SFT_DATA" --input-key messages --tool-key tools
  --rollout-shuffle --rollout-seed 42 --num-epoch 2
  --rollout-batch-size 32 --global-batch-size 32 --n-samples-per-prompt 1
  --loss-type sft_loss --loss-mask-type qwen3_5 --calculate-per-token-loss
  --disable-compute-advantages-and-returns --debug-train-only --log-probs-chunk-size 4096
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size 4 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --seq-length 131072 --max-position-embeddings 131072 --use-dynamic-batch-size --max-tokens-per-gpu 32768
  --optimizer adam --lr 5e-6 --min-lr 5e-7 --lr-decay-style cosine --lr-warmup-fraction .03
  --weight-decay .1 --adam-beta1 .9 --adam-beta2 .95 --adam-eps 1e-8 --clip-grad 1
  --use-distributed-optimizer --attention-dropout 0 --hidden-dropout 0
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
)
if [[ "${RESUME:-0}" == 1 ]]; then
  test -f "$SAVE_DIR/latest_checkpointed_iteration.txt"
  TRAIN_ARGS+=(--load "$SAVE_DIR")
elif [[ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Existing checkpoint: set RESUME=1 to resume explicitly." >&2; exit 2
fi
printf '%q ' python "$REPO_DIR/training/backend/train_async.py" "${TRAIN_ARGS[@]}"
printf '\n'
ray start --head --node-ip-address=127.0.0.1 --num-gpus=8 --port=6380 --dashboard-host=127.0.0.1 --dashboard-port=8265 --disable-usage-stats
ray status --address=127.0.0.1:6380
export RAY_ADDRESS=127.0.0.1:6380
exec python -u "$REPO_DIR/training/backend/train_async.py" "${TRAIN_ARGS[@]}"
