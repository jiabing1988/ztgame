import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

logger = get_logger()
seed_everything(42)

# 模型
model_id_or_path = 'Qwen/Qwen2.5-3B-Instruct' 
system = 'You are a helpful assistant.'
output_dir = 'output'

# 数据集
dataset = ['AI-ModelScope/alpaca-gpt4-data-zh#500', 'AI-ModelScope/alpaca-gpt4-data-en#500',
           'swift/self-cognition#500'] 
data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01  # 切分验证集
num_proc = 4  # 预处理的进程数
# 替换自我认知数据集中的填充符：{{NAME}}, {{AUTHOR}}
model_name = ['小黄', 'Xiao Huang']  # 模型的中文名和英文名
model_author = ['魔搭', 'ModelScope']  # 模型作者的中文名和英文名

lora_rank = 8
lora_alpha = 32

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')