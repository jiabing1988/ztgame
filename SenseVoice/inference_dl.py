import os
import json
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from tqdm import tqdm

model_dir = "iic/SenseVoiceSmall"
model = AutoModel(model=model_dir, trust_remote_code=True, device="cuda:2")

input_jsonl = "/ailab-train/speech/dingli/fenli_vocals_test_1000.jsonl"  
output_jsonl = "/ailab-train/speech/dingli/sensevoice/SenseVoice/output/fenli_vocals_test_1000_output.jsonl"  

audio_paths = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())  
        if "path" in data:
            audio_paths.append(data["path"])  

if len(audio_paths) == 0:
    raise ValueError("音频路径文件为空，请检查！")

with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for audio_path in tqdm(audio_paths, desc="Processing Audio Files"):
        if not os.path.exists(audio_path):
            print(f"警告：文件 {audio_path} 不存在，跳过")
            continue

        try:
            res = model.generate(
                input=audio_path,
                cache={},
                language="auto", 
                use_itn=True,
                batch_size=64, 
            )

            text = rich_transcription_postprocess(res[0]["text"])

            output_data = {
                # "key": os.path.basename(audio_path),  
                "key": audio_path.split('/')[-2],  
                "text": text  
            }

            json.dump(output_data, f_out, ensure_ascii=False)
            f_out.write("\n")

            print(f"处理完成: {audio_path}")

        except Exception as e:
            print(f"处理 {audio_path} 时出错: {e}")

print(f"\n所有转录结果已保存到 {output_jsonl}")

