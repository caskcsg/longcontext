#!/bin/bash -l
#SBATCH -J train_64K
#SBATCH -N 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -c 32

# !!!! Load your own environment here !!!! #
# !!!! Load your own environment here !!!! #
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

apt update && apt install iproute2 
pip install wandb


# Fine-tune from this model 

model=${MODEL:-"your 512k base model path"}
# Point to the base dir of the ProLong 64K data
dataset=${DATASET:-"your sft data path"}

# Directories in the dataset root folder where @ is followed by the mixing proportion 
domains=(
    NExtLong-Instruct-dataset-Magpie-Llama-3.3-Pro-1M-v0.1@1.0
)

domains_name=LiteLong_NExtLong_512K_Magpie


# shift_rank=${shift_rank:-${RANK}}

shift_rank=$RANK


seq_parallel_size=${SEQ_PARALLEL_SIZE:-8}


bsz=${BSZ:-128} # * 131k * 1 acc *  128 gpus  / 8 = 2M
seq=${SEQ:-1} # per-device batch size
lr=${LR:-2e-5}
steps=${STEPS:-250} # 2M * 250  = 500M
save_steps=${SAVE:-500}
warmup=${WARMUP:-0.05}
suffix=${SUFFIX:-""} # for model saving name

run_name="sft_$(basename $model)_${domains_name}_sp${seq_parallel_size}_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}${suffix}"

out_dir="checkpoints/$run_name"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

num_nodes=$WORLD_SIZE

# Gradient accumulation
accu=$(($bsz / $seq / $num_gpus / $num_nodes ))
# acc 为1 



# [0] Disable
# [1] FULL_SHARD (shards optimizer states, gradients and parameters),
# [2] SHARD_GRAD_OP (shards optimizer states and gradients),
# [3] NO_SHARD (DDP),
# [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy),
# [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy). For more information, please refer the official PyTorch docs.
fsdp=${FSDP:-"1"}
gc=${GC:-"1"}

export LOGIT_BLOCK_SIZE=2048  # Compute Llama logits in blocks of 2048 tokens

mkdir -p $out_dir
nvidia-smi

if [ $num_nodes -gt 1 ]; then
    master_addr=${MASTER_ADDR}

    # Launch via srun
    header="torchrun \
    --master_addr $master_addr\
    --master_port 56321 \
    --nnodes=$num_nodes \
    --node_rank $shift_rank \
    --nproc_per_node=$num_gpus \
    -m training.train_language_model"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
 
    # Launch without srun
    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
fi
echo "slurm_nodelist=${SLURM_NODELIST} num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

export OMP_NUM_THREADS=$num_gpus
export WANDB_PROJECT="prolong"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline" # We turn off wandb online sync by default
export TOKENIZERS_PARALLELISM=true


base_arguments=(
    --report_to wandb
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --config_overrides_json "$overrides"
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    --bf16
    --learning_rate $lr
    --min_lr_ratio 0.1
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --dataloader_num_workers 1

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --per_device_max_tokens 131072

    --cuda_empty_cache

    # --apply_instruct_masks # mask out the tokens from instructions (instead of responses) when calculating losses
    # --token_scaled_loss # average losses over valid training tokens instead of devices

    --seq_parallel_size ${seq_parallel_size}
)



if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp 
    base_arguments+=( 
        --fsdp "full_shard auto_wrap"
        --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
        --fsdp_config '{"offload_param": true, "offload_optimizer": true, "limit_all_gathers": true, "forward_prefetch": true, "backward_prefetch": "backward_pre", "activation_checkpointing": true, "cpu_offload": true, "activation_cpu_offload": true, "mixed_precision": true}'
    )
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

# if [ $gc -ne 0 ]; then
#     base_arguments+=( --gradient_checkpointing )
# fi

base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log_${shift_rank}.out

wait