以下是符合规范的 `README.md` 文件内容：

````markdown
# 音频分析模型集合

## 目录结构

本项目包含多个音频分析模型，具体包括以下模块：

1. **Qwen2-Audio**: 用于音频推理的模型。
2. **Mellow**: 用于音频分析的模型。
3. **Sensevoice**: 音频分析模型，进行ASR推理。
4. **Whisper-large-v3**: ASR模型，用于语音转录。
5. **InternVL**: 计算音频分析模型输出的回答与标注回答相似性的代码。
6. **Finetune-Qwen2Audio**: 微调Qwen2-Audio模型的代码，包括全量微调和LoRA微调的脚本。

## 模块详细说明

### 1. Qwen2-Audio

此模块包含用于音频推理的代码，执行以下命令进行推理：

```bash
python inference_dl.py
````

### 2. Mellow

此模块为音频分析模型，提供多个示例脚本，任意执行以下一个即可：

```bash
python example.py
python example1.py
python example2.py
python example3.py
```

### 3. Sensevoice

此模块用于进行ASR推理，运行以下命令进行推理：

```bash
python inference_dl.py
```

### 4. Whisper-large-v3

该模块是一个ASR模型，用于语音转录，执行以下命令进行推理：

```bash
python inference_dl.py
```

### 5. InternVL

此模块包含计算音频分析模型输出的回答与标注的回答是否相似的代码。

### 6. Finetune-Qwen2Audio

此模块用于微调 Qwen2-Audio 模型，包含以下两种微调方法：

* **全量微调**: 执行 `train_full.sh` 脚本。
* **LoRA微调**: 执行 `train_lora.sh` 脚本。

## 运行环境

确保已安装以下依赖包：

* Python 3.x
* 其他依赖可以通过运行 `requirements.txt` 文件安装：

```bash
pip install -r requirements.txt
```

## 注意事项

1. 请根据需要修改代码中的路径和配置文件。
2. 对于每个模型，确保输入的数据格式符合要求。
3. 根据硬件配置，可能需要调整批处理大小（batch size）和其他超参数。
