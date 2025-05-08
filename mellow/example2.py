import torch
import json
from pathlib import Path
import os
from mellow import MellowWrapper

def read_audio_paths(jsonl_file):
    """ 读取 jsonl 文件中的音频路径 """
    paths = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())  
            if "path" in data:
                paths.append(data["path"]) 
    return paths

if __name__ == "__main__":
    # setup cuda and device
    cuda = torch.cuda.is_available()
    device = 0 if cuda else "cpu"

    # setup mellow
    mellow = MellowWrapper(
        config="v0",
        model="v0",
        device=device,
        use_cuda=cuda,
    )

    jsonl_file = "/ailab-train/speech/dingli/test_wav.jsonl"
    output_file = "/ailab-train/speech/dingli/mellow/outputs/output_no_format_test_people.jsonl"

    audio_paths = read_audio_paths(jsonl_file)

    if len(audio_paths) < 1:
        raise ValueError("音频路径文件为空，请检查！")

    results = []

    for path in audio_paths:
        # prompt = (
        #     "What sound effects are in this audio? "
        #     "Please answer in the following format: \n"
        #     "a) [First sound effect]\n"
        #     "b) [Second sound effect]\n"
        #     "c) [Third sound effect]\n"
        #     "Provide a brief description for each."
        # )
        prompt = "Does this audio contain human voices? Please answer yes or no."

        examples = [[path, path, prompt]]
        response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)

        results.append({"path": path, "output": response})

        print(f"Processed: {path}", f"Response: {response}")

    with open(output_file, "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n所有音频处理完成，结果已保存到: {output_file}")