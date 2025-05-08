import json
import os

input_jsonl_file = "/user-fs/chenzihao/dingli/FSD50K_effect/processed_labels/all_train_labels_v1.jsonl"  
output_jsonl_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/data/train.jsonl"
audio_folder = "/user-fs/chenzihao/dingli/music_effect_combined"  

output_data = []

with open(input_jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        # 构造 query, response, audios
        query = "<audio>What are the sound effects of this audio?" 
        response = data['label']
        audio_file = data['audio_file']
        audio_path = os.path.join(audio_folder, audio_file)

        entry = {
            "query": query,
            "response": response,
            "audios": [audio_path]
        }
        
        output_data.append(entry)
with open(output_jsonl_file, 'w', encoding='utf-8') as f:
    for entry in output_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print("✅ JSONL 文件已生成！")
