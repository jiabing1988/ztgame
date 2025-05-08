import json
import os
import librosa
from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from torch import device

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
output_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/output/test_1000_320_ola_novideo.jsonl"

# prompt_text = (
#     "请分析这段音频中包含了哪些音效，并按它们在音频中出现的先后顺序列出关键词。"
#     "回答时只需列出每个音效的关键词，保持简洁，不要添加额外说明。"
# )

# prompt_text = (
#     "Please analyze the audio and list all the sound effects it contains in the order they appear. "
#     "When responding, only list the keywords for each sound effect, keep it concise, and do not add any extra explanation. "
#     "Format your response like this: "
#     "1. <keyword1>\n"
#     "2. <keyword2>\n"
#     "3. <keyword3>\n"
#     "... (continue listing all sound effects as needed)"
# )

prompt_text = (
    "What are the sound effects of this audio?"
)

results = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        audio_path = item["path"]

        audio = load_audio(audio_path)
        if audio is None:
            continue

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt_text},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=[text], audios=[audio], return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to("cuda")

        generate_ids = model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"path: {audio_path}")
        print(f"response: {response}")

        result = {
            "path": audio_path,
            "prompt": prompt_text,
            "output": response
        }
        results.append(result)

with open(output_file, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"Transcriptions saved to {output_file}")
