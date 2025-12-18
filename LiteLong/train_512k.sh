#!/bin/bash -l
#SBATCH -J train_512K
#SBATCH -N 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -c 32

# !!!! Load your own environment here !!!! #
# !!!! Load your own environment here !!!! #

apt update && apt install iproute2 
pip install wandb



id_name=${id_name:-"cosmopedia_fineweb55"}


input_id_name=${id_name}


# Fine-tune from this model 

model=${MODEL:-"your llama3-8b-instruct model path"}

# Point to the base dir of the ProLong 512K data

dataset=${DATASET:-"your 512k prolong format data path"}

new_domains=(
    litelong_nextlong_512k@1
)



domains_name=LiteLong_NExtLong_512K


bsz=${BSZ:-128} # * 512K (seq len) / 8 (seq parallel size) = 8M
seq=${SEQ:-1} # per-device batch size
lr=${LR:-5e-6}
steps=${STEPS:-500}
save_steps=${SAVE:-25}
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-""} # for model saving name


run_name="${input_id_name}_512k"

out_dir="checkpoints/$run_name"

model_64k_dir=${model_64k_dir:-"your 64k checkpoint path"}

declare -a checkpoints
for dir in "$out_dir"/checkpoint-*; do
    if [ -d "$dir" ]; then
        # 提取数字部分
        num=$(basename "$dir" | sed 's/checkpoint-//')
        if [[ $num =~ ^[0-9]+$ ]]; then
            checkpoints+=("$num")
        fi
    fi
done

is_ok_continue="0"

# 判断数组是否为空
if [ ${#checkpoints[@]} -gt 0 ]; then
    # 找到最大的数字
    max_num=$(printf "%s\n" "${checkpoints[@]}" | sort -nr | head -n 1)
    echo "最大的 checkpoint- 数字是: $max_num"
    
    resume_ckpt="$out_dir/checkpoint-$max_num"

    resume_ckpt_safetensors="$resume_ckpt/model-00007-of-00007.safetensors"

    if [ -f "$resume_ckpt_safetensors" ]; then
        echo "存在 $resume_ckpt_safetensors，可以继续训练。"
        is_ok_continue="1"
    else
        echo "不存在 $resume_ckpt_safetensors，无法继续训练。"
        
    fi

    echo "从 ${resume_ckpt} 进行加载"
fi


if [ "$is_ok_continue" == "0" ]; then
    mkdir -p "$out_dir/checkpoint-0"
    resume_ckpt="$out_dir/checkpoint-0"
    echo "没有找到 'checkpoint-数字' 文件夹，已创建 'checkpoint-0'."
    
    # 如果没有找到, 创建 checkpoint-0 文件夹
    if [ "$RANK" -eq 0 ]; then


        mkdir -p ${resume_ckpt}


        cp -r ${model_64k_dir}/*json ${resume_ckpt}/

        echo "copy json from ${model_64k_dir}"
        cp -r ${model_64k_dir}/*safetensors ${resume_ckpt}/

        echo "copy safetensors from ${model_64k_dir}"
        cp -r ${model_64k_dir}/*bin ${resume_ckpt}/

        echo "copy bin from ${model_64k_dir}"
        rm ${resume_ckpt}/streaming_dataset_state.json

        echo "copy done"
        
    else
        sleep 2m
    fi

fi
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

num_nodes=$WORLD_SIZE
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi

num_nodes=${WORLD_SIZE:-$num_nodes}

# Gradient accumulation
accu=$(($bsz / $seq / $num_gpus / $num_nodes))


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
    # master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR}

    # Launch via srun
    header="torchrun \
    --master_addr $master_addr\
    --master_port 56321 \
    --nnodes=$num_nodes \
    --node_rank $RANK \
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

export GLOO_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1
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

    # Initialize model + optimizer state with ProLong64K
    --resume_from_checkpoint $resume_ckpt
    
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

    --per_device_max_tokens 524288

    # --torch_compile
    --cuda_empty_cache
    --config_overrides "rope_theta=128000000"

    --seq_parallel_size 8
)



if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp 
    base_arguments+=( --fsdp "auto_wrap" )
    # [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

base_arguments+=( --tokenized_mds_train )

for domain in "${new_domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done


echo command: "${header} ${base_arguments[@]}"

${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out