import torch
import json
from pathlib import Path
import os
from mellow import MellowWrapper

def read_audio_paths(jsonl_file):
    """Read audio paths from jsonl file"""
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
    output_file = "/ailab-train/speech/dingli/mellow/outputs/output_musicemotion_prompt2.jsonl"

    audio_paths = read_audio_paths(jsonl_file)

    if len(audio_paths) < 1:
        raise ValueError("The audio path list is empty. Please check the input file.")

    prompt = "Please describe the overall musical emotion of this audio clip using a simple keyword or short phrase. Only consider the first audio file."

    results = []

    for path in audio_paths:
        examples = [[path, path, prompt]] 
        response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)

        result = {
            "path": path,
            "prompt": prompt,
            "output": response
        }
        results.append(result)

        print(f"Processed: {path}")
        print(f"Response: {response}")

    with open(output_file, "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nAll audio processed. Results saved to: {output_file}")
