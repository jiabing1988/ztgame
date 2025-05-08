import json
import os
import librosa
from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("/user-fs/chenzihao/dingli/finetune_qwen2audio/output/qwen2-audio-7b-instruct/v0-20250421-185057/checkpoint-1400-merged")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "/user-fs/chenzihao/dingli/finetune_qwen2audio/output/qwen2-audio-7b-instruct/v0-20250421-185057/checkpoint-1400-merged", device_map="auto"
)

def load_audio(audio_path):
    if audio_path.startswith("http"):
        audio, _ = librosa.load(BytesIO(urlopen(audio_path).read()), sr=processor.feature_extractor.sampling_rate)
    else:
        if not os.path.exists(audio_path):
            print(f"音频文件不存在，跳过：{audio_path}")
            return None
        audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    return audio

input_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/data/test1000_320.jsonl"
output_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/output/test_1000_320_Videollama3 .jsonl"
video_description_file = "/user-fs/chenzihao/liangyunming/VLM/VideoLLaMA3/res_test1000_320_rename.txt" 

video_descriptions = {}
with open(video_description_file, "r", encoding="utf-8") as f:
    for line in f:
        video_path, description = line.strip().split('|', 1)
        video_descriptions[os.path.basename(video_path)] = description.strip()  

prompt_text = "Combined with this video description, What are the sound effects of this audio?"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        audio_path = item["path"]
        
        video_file_name = os.path.basename(audio_path).replace(".wav", ".mp4").replace(".mp3", ".mp4")
        video_description = video_descriptions.get(video_file_name, "No description available.")
        
        full_prompt = f"{prompt_text} Video description: {video_description}"
        audio = load_audio(audio_path)
        if audio is None:
            continue
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": full_prompt}, 
            ]}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=[text], audios=[audio], return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to("cuda")

        generate_ids = model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        result = {
            "path": audio_path,
            "prompt": full_prompt,
            "output": response
        }
        results.append(result)

with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Transcriptions saved to {output_file}")
