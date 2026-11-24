#!/usr/bin/env bash
set -Eeuo pipefail
set -x
: "${NEW_OPD_LOG_ACTIVE:?Use run_train.sh so this command is logged}"

# MedKAD launch configuration using the local verl runtime.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_SOURCE_DIR="${REPO_DIR}"
CODE_DIR="$(dirname "${REPO_DIR}")"
WORKSPACE_DIR="$(dirname "${CODE_DIR}")"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/miniconda3/envs/verl}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}"
cd "$REPO_DIR"

: "${TRAIN_DATASET:?Task launcher must set TRAIN_DATASET}"
: "${TEST_DATASET_PATH:?Task launcher must set TEST_DATASET_PATH}"
: "${TRAIN_DATASET_NAME:?Task launcher must set TRAIN_DATASET_NAME}"
: "${TASK_DISPLAY_NAME:?Task launcher must set TASK_DISPLAY_NAME}"

export TEST_DATASET="${TEST_DATASET:-[\"${TEST_DATASET_PATH}\"]}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${CODE_DIR}/model/Qwen3-VL-2B-Instruct}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${CODE_DIR}/model/Qwen3-VL-4B-Instruct}"

[[ -f "$TRAIN_DATASET" ]] || { echo "Training dataset not found: $TRAIN_DATASET" >&2; exit 1; }
[[ -f "$TEST_DATASET_PATH" ]] || { echo "Test dataset not found: $TEST_DATASET_PATH" >&2; exit 1; }
[[ -d "$ACTOR_MODEL_PATH" ]] || { echo "Student model not found: $ACTOR_MODEL_PATH" >&2; exit 1; }
[[ -d "$REWARD_MODEL_PATH" ]] || { echo "Teacher model not found: $REWARD_MODEL_PATH" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }

# This adapter keeps the original verl/FSDP/vLLM path and accepts an explicit
# visible-GPU count from the task launcher.
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
unset FLASH_ATTENTION_TRITON_AMD_ENABLE VLLM_USE_TRITON_FLASH_ATTN
unset ROCM_HOME ROCM_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
[[ "$TRAINER_N_GPUS_PER_NODE" =~ ^[1248]$ ]] || {
    echo "TRAINER_N_GPUS_PER_NODE must be 1, 2, 4 or 8; got: $TRAINER_N_GPUS_PER_NODE" >&2
    exit 2
}
export NUM_GPUS="$TRAINER_N_GPUS_PER_NODE"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${VERL_SOURCE_DIR}/verl:${PYTHONPATH:-}"

export CACHE_ROOT="${CACHE_ROOT:-${CODE_DIR}/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${CACHE_ROOT}/outlines/opd_${TRAIN_DATASET_NAME}_$$}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
mkdir -p "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR"

# The experiment launcher selects the objective; existing runs default to OPD.
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-STANDARD_OPD}"
[[ "$ADV_ESTIMATOR" == STANDARD_OPD || "$ADV_ESTIMATOR" == KALIGN_OPD ]] || {
  echo "Unsupported ADV_ESTIMATOR=$ADV_ESTIMATOR" >&2; exit 2;
}

# The 4B teacher receives a privileged GT contrastive hint by default; the 2B
# student rollout prompt remains unchanged. These can be overridden at launch.
export INJECT_MODE="${INJECT_MODE:-gt_contrastive}"
export GT_INJECT_SCHEDULE="${GT_INJECT_SCHEDULE:-constant}"
export GT_INJECT_RATIO="${GT_INJECT_RATIO:-1.0}"
export GT_INJECT_START="${GT_INJECT_START:-1.0}"
export GT_INJECT_END="${GT_INJECT_END:-0.0}"
export GT_INJECT_ANNEAL_STEPS="${GT_INJECT_ANNEAL_STEPS:-100}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-2560}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-2560}"
export MIN_RESPONSE_TOKENS="${MIN_RESPONSE_TOKENS:-0}"
export MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + (MAX_RESP_LENGTH > MAX_VAL_RESP_LENGTH ? MAX_RESP_LENGTH : MAX_VAL_RESP_LENGTH)))

export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export REWARD_MICRO_BATCH_SIZE="${REWARD_MICRO_BATCH_SIZE:-1}"
export N_RESPONSES="${N_RESPONSES:-8}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
[[ "${TOP_K_STRATEGY:-only_stu}" == only_stu ]] || { echo "MedKAD requires TOP_K_STRATEGY=only_stu" >&2; exit 2; }
[[ "${REWARD_WEIGHT_MODE:-student_p}" == student_p ]] || { echo "MedKAD requires REWARD_WEIGHT_MODE=student_p" >&2; exit 2; }
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export USE_COT="${USE_COT:-True}"

export MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12888}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"

# The clean-OPD format1 arm adds the historical sampled-token format loss.
export ENABLE_FORMAT_REWARD="${ENABLE_FORMAT_REWARD:-False}"
export FORMAT_LOSS_WEIGHT="${FORMAT_LOSS_WEIGHT:-0.0}"
if [[ "$FORMAT_LOSS_WEIGHT" != 0.0 && "$FORMAT_LOSS_WEIGHT" != 0 ]]; then
    [[ "$ENABLE_FORMAT_REWARD" == True ]] || {
        echo "FORMAT_LOSS_WEIGHT requires ENABLE_FORMAT_REWARD=True" >&2; exit 2;
    }
fi
[[ "${ENABLE_THINK_REWARD:-False}" == False && "${THINK_LOSS_WEIGHT:-0.0}" == 0.0 ]] || { echo "MedKAD does not use think loss" >&2; exit 2; }
export ENABLE_THINK_REWARD=False THINK_LOSS_WEIGHT=0.0

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-20}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export VAL_N="${VAL_N:-3}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-5}"

export PROJECT_NAME="${PROJECT_NAME:-MedKAD}"
export USE_SWANLAB="${USE_SWANLAB:-True}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
if [[ "$USE_SWANLAB" == "True" ]]; then
    TRAINER_LOGGER="['console','swanlab']"
else
    TRAINER_LOGGER="['console']"
fi

ACTOR_MODEL_NAME="$(basename "$ACTOR_MODEL_PATH")"
REWARD_MODEL_NAME="$(basename "$REWARD_MODEL_PATH")"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_NAME="${RUN_NAME:-MedKAD_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_teacher_${REWARD_MODEL_NAME}_topk_${LOG_PROB_TOP_K}-hint_${INJECT_MODE}_${GT_INJECT_SCHEDULE}_${GT_INJECT_RATIO}-${RUN_TIMESTAMP}}"

# Set RESUME_FROM to an existing run directory to preserve its run name and
# make trainer.resume_mode=auto load the step recorded by its tracker file.
export RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_FROM="$(realpath -e "$RESUME_FROM")"
    TRACKER_FILE="${RESUME_FROM}/latest_checkpointed_iteration.txt"
    [[ -f "$TRACKER_FILE" ]] || {
        echo "Resume tracker not found: $TRACKER_FILE" >&2
        exit 1
    }
    RESUME_STEP="$(tr -d '[:space:]' < "$TRACKER_FILE")"
    [[ "$RESUME_STEP" =~ ^[0-9]+$ ]] || {
        echo "Invalid resume step in $TRACKER_FILE: $RESUME_STEP" >&2
        exit 1
    }
    [[ -d "${RESUME_FROM}/global_step_${RESUME_STEP}/actor" && -f "${RESUME_FROM}/global_step_${RESUME_STEP}/data.pt" ]] || {
        echo "Checkpoint for step $RESUME_STEP is incomplete: $RESUME_FROM" >&2
        exit 1
    }
    RUN_NAME="$(basename "$RESUME_FROM")"
    CKPT_PATH="$RESUME_FROM"
    echo "Resuming run '$RUN_NAME' from global_step_${RESUME_STEP}"
else
    export CKPT_PATH="${CKPT_PATH:-${REPO_DIR}/checkpoints/${RUN_NAME}}"
fi
export RUN_NAME CKPT_PATH
export LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
export VALIDATION_DIR="${VALIDATION_DIR:-${REPO_DIR}/validation_log/${RUN_NAME}}"
mkdir -p "$LOG_DIR" "$CKPT_PATH" "$VALIDATION_DIR"

# The foreground task launcher owns the complete log via tee.
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
ulimit -n 1048576 || true

export RAY_PORT="${RAY_PORT:?Task launcher must set a unique RAY_PORT}"
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
# Ray appends session_*/sockets/plasma_store to this path. Keep the prefix
# short enough for Linux's 107-byte AF_UNIX socket-path limit, and unique per
# launcher PID so cleanup never targets another run.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rno_${RAY_PORT}_$$}"
if (( ${#RAY_TEMP_DIR} > 40 )); then
    echo "Ray temporary path is too long for Unix sockets: $RAY_TEMP_DIR" >&2
    exit 2
fi
export RAY_OBJECT_STORE_MEMORY_BYTES="${RAY_OBJECT_STORE_MEMORY_BYTES:-1073741824}"

cleanup_local_ray_instance() {
    local ray_pids=()
    mapfile -t ray_pids < <(
        ps -eo pid=,args= | awk -v marker="${RAY_TEMP_DIR}/" \
            'index($0, marker) && index($0, "awk -v marker=") == 0 { print $1 }'
    )
    if (( ${#ray_pids[@]} > 0 )); then
        echo "Stopping Ray processes for ${RAY_TEMP_DIR}: ${ray_pids[*]}"
        kill "${ray_pids[@]}" 2>/dev/null || true
        sleep 3
        mapfile -t ray_pids < <(
            ps -eo pid=,args= | awk -v marker="${RAY_TEMP_DIR}/" \
                'index($0, marker) && index($0, "awk -v marker=") == 0 { print $1 }'
        )
        if (( ${#ray_pids[@]} > 0 )); then
            kill -9 "${ray_pids[@]}" 2>/dev/null || true
        fi
    fi
}

echo "============================================================"
echo "$ADV_ESTIMATOR: $TASK_DISPLAY_NAME"
echo "Student:      $ACTOR_MODEL_PATH"
echo "Teacher:      $REWARD_MODEL_PATH"
echo "Hint:         $INJECT_MODE ($GT_INJECT_SCHEDULE, ratio=$GT_INJECT_RATIO)"
echo "CUDA:         $CUDA_VISIBLE_DEVICES"
echo "SwanLab:      $PROJECT_NAME"
echo "Checkpoint:   $CKPT_PATH"
echo "Log:          $LOG_FILE"
echo "============================================================"

IFS=',' read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#cuda_devices[@]} != TRAINER_N_GPUS_PER_NODE )); then
    echo "Expected $TRAINER_N_GPUS_PER_NODE CUDA device ID(s); got: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

if [[ "${ALLOW_CONCURRENT_TRAINING:-False}" != "True" ]]; then
    mapfile -t existing_trainers < <(pgrep -af "[v]erl\.trainer\.main_ppo" || true)
    if (( ${#existing_trainers[@]} > 0 )); then
        echo "Another verl training process is already running:" >&2
        printf '  %s\n' "${existing_trainers[@]}" >&2
        exit 1
    fi
fi

[[ ! -e "$RAY_TEMP_DIR" ]] || {
    echo "Ray temporary directory already exists; refusing to take ownership: $RAY_TEMP_DIR" >&2
    exit 1
}
mkdir -p "$RAY_TEMP_DIR"
trap cleanup_local_ray_instance EXIT

if ! "$PYTHON_BIN" -c 'import ray.dashboard.agent'; then
    echo "WARNING: optional ray.dashboard.agent import failed; continuing with --include-dashboard=false" >&2
fi
export RAY_agent_register_timeout_ms=120000
ray start --head --num-cpus="${RAY_NUM_CPUS:-8}" --num-gpus="$TRAINER_N_GPUS_PER_NODE" \
    --object-store-memory="$RAY_OBJECT_STORE_MEMORY_BYTES" \
    --port="$RAY_PORT" \
    --include-dashboard=false \
    --disable-usage-stats \
    --system-config='{"worker_register_timeout_seconds":300,"agent_register_timeout_ms":120000}' \
    --temp-dir="$RAY_TEMP_DIR"
sleep 5
ray status --address="$RAY_ADDRESS"

TRAINING_STEP_ARGS=()
if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
    TRAINING_STEP_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi
WARMUP_ARGS=("actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO")
if [[ -n "$LR_WARMUP_STEPS" ]]; then
    WARMUP_ARGS=("actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS")
fi

HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${REPO_DIR}/outputs/hydra/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)_pid$$}"
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    data.shuffle=True \
    data.seed=42 \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size="$MINI_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESP_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.image_key=images \
    +data.use_cot="$USE_COT" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    "${WARMUP_ARGS[@]}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.n="$N_RESPONSES" \
    +actor_rollout_ref.rollout.seed="$EXPERIMENT_SEED" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.log_prob_top_k="$LOG_PROB_TOP_K" \
    +actor_rollout_ref.rollout.top_k_strategy="$TOP_K_STRATEGY" \
    +actor_rollout_ref.rollout.reward_weight_mode="$REWARD_WEIGHT_MODE" \
    +actor_rollout_ref.rollout.teacher_temperature="$TEACHER_TEMPERATURE" \
    actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
    +actor_rollout_ref.rollout.min_tokens="$MIN_RESPONSE_TOKENS" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="$MAX_VAL_RESP_LENGTH" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.format_max_rounds=5 \
    +actor_rollout_ref.rollout.val_kwargs.min_tokens="$MIN_RESPONSE_TOKENS" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="$MODEL_DTYPE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    reward_model.enable=True \
    +reward_model.inject_mode="$INJECT_MODE" \
    +reward_model.reward_kwargs.enable_format_reward="$ENABLE_FORMAT_REWARD" \
    +reward_model.reward_kwargs.enable_think_reward="$ENABLE_THINK_REWARD" \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.dtype="$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu="$REWARD_MICRO_BATCH_SIZE" \
    custom_reward_function.path="${VERL_SOURCE_DIR}/verl/verl/utils/reward_score/medical_vqa/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.resume_mode=auto \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.log_val_generations=2 \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.validation_data_dir="$VALIDATION_DIR" \
    trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "${TRAINING_STEP_ARGS[@]}" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=False

echo "MedKAD training finished: $(date)"
