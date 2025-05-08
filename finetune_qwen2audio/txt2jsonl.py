import json

input_txt_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/data/test1000_320_rename _wav.txt"  
output_jsonl_file = "/user-fs/chenzihao/dingli/finetune_qwen2audio/data/test1000_320.jsonl" 

with open(input_txt_file, 'r') as infile:
    with open(output_jsonl_file, 'w') as outfile:
        for line in infile:
            line = line.strip()  
            if line: 
                result_entry = {"path": line}
                json.dump(result_entry, outfile, ensure_ascii=False)
                outfile.write('\n')  

print(f"✅ JSONL 文件已生成，保存在 {output_jsonl_file}")
