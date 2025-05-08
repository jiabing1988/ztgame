import os
import torch
import json
import torchaudio
from tqdm import tqdm
from modelscope import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/ailab-train/speech/dingli/models/AI-ModelScope/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

input_jsonl = "/ailab-train/speech/liangyunming/20250212/audio-flamingo/inference/aa.jsonl"  
output_jsonl = "/ailab-train/speech/dingli/whisper-large-v3/outputs/whisper_large_v3_output.jsonl"  

audio_paths = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())  
        if "path" in data:
            audio_paths.append(data["path"])  

if len(audio_paths) == 0:
    raise ValueError("音频路径文件为空，请检查！")

MAX_DURATION = 30  
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for audio_path in tqdm(audio_paths, desc="Processing Audio Files"):
        if not os.path.exists(audio_path):
            print(f"警告：文件 {audio_path} 不存在，跳过")
            continue
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            total_duration = waveform.shape[1] / sample_rate
            if total_duration > MAX_DURATION:
                waveform = waveform[:, :MAX_DURATION * sample_rate]
                print(f"音频 {audio_path} 超过 {MAX_DURATION} 秒，已截取前 {MAX_DURATION} 秒")

            temp_path = audio_path.replace(".wav", "_temp.wav")
            torchaudio.save(temp_path, waveform, sample_rate)

            result = pipe(temp_path)

            os.remove(temp_path)
            output_data = {
                "key": os.path.basename(audio_path),
                "text": result["text"]
            }

            json.dump(output_data, f_out, ensure_ascii=False)
            f_out.write("\n")

            print(f"处理完成: {audio_path}")

        except Exception as e:
            print(f"处理 {audio_path} 时出错: {e}")

print(f"\n转录结果已保存到 {output_jsonl}")
