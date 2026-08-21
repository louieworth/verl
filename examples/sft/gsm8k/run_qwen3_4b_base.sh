: "${WANDB_API_KEY:?Set WANDB_API_KEY in the runtime environment}"
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=4
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/data/data/jiangli/huggingface/datasets/gsm8k/train.parquet \
    data.val_files=/data/data/jiangli/huggingface/datasets/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=64 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3-8b-instruct \
    trainer.logger=console \
    trainer.total_epochs=2 $@ \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_4b_instruct_function_rm' \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.device=npu
