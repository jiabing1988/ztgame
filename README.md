# ztgame交接代码

1、Qwen2-Audio里面是Qwen2-Audio的推理代码，运行python inference_dl.py即可
2、mellow也是音频分析模型的代码，运行example.py、example1.py、example2.py、example3.py任意一个都可以
3、Sensevoice也是音频分析模型，运行inference_dl.py即可进行ASR推理
4、Whisper-large-v3是一个ASR模型，运行inference_dl.py即可进行语音转录
5、InternVL里面包含一个计算这些音频分析模型的回答与标注的回答是否相似的代码
6、finetune_qwen2audo里面包含微调qwen2-audio的代码，其中有全量微调和Lora微调的脚本，全量微调的脚本是train_full.sh，lora微调的脚本时train_lora.sh。
