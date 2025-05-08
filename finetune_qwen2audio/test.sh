CUDA_VISIBLE_DEVICES=0 swift infer \
  --ckpt_dir /user-fs/chenzihao/dingli/finetune_qwen2audio/output/qwen2-audio-7b-instruct/v0-20250421-185057/checkpoint-1400 \
  --load_dataset_config true --merge_lora true