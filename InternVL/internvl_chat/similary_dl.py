import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


if __name__ == "__main__":
    # path = "/ailab-train/speech/zhengjunjie/code/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_qa-merge"
    path = "/ckptstorage/chenzihao/zhengjunjie/opt/huggingface/InternVL-2B"

    model = InternVLChatModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=128, do_sample=False, temperature=0.7)

    t1 = time.time()
    # ans = "The music in this video is instrumental classical piano. The mood of the piece is melancholic and somber, with a slow tempo that creates a sense of introspection and reflection. The piano plays a gentle melody that evokes feelings of sadness and longing. The piece has a haunting quality to it, with the piano's notes echoing in the listener's mind. Overall, the music is beautiful and poignant, capturing the emotional weight of the scene."
    # question = f"Summarize five main keywords for the music mood from: {ans}. Format the output as:\n1. <keyword>\n2. <keyword>\n3. <keyword>"

    # infile = "/ailab-train/speech/liangyunming/20250212/Ola/test_videos/res_clips_changjinghu_more3s_music_type.txt"
    # outfile = "/ailab-train/speech/liangyunming/20250212/Ola/test_videos/res_clips_changjinghu_more3s_music_type_keywords.txt"
    infile = "//user-fs/chenzihao/dingli/InternVL/process_dl/processed_output/music_effect_output.jsonl"
    outfile = "/user-fs/chenzihao/dingli/InternVL/process_dl/similary_output/music_effect_similary.jsonl"
    
    with open(infile, "r", encoding='utf-8') as fr, open(outfile, "w", encoding='utf-8') as fw:
        for line in tqdm(fr.readlines()):
            data = json.loads(line.strip())
            key = data['key']
            output1 = data['output1']
            output2 = data['output2']
            # output3 = data['output3']
            
            question = f"Do Output 1 and Output 2 express similar concepts or include similar elements? Respond with 'Yes' if they convey similar meanings or ideas, 'No' if they do not, or 'Maybe' if it's unclear.\n\nOutput 1: {output1}\nOutput 2: {output2}"
            # question = f"""
            # Given output1, output2, and output3, what is the common sound effect they are expressing? 
            # Please summarize the sound effect that is shared by these three outputs.

            # Output 1: {output1}
            # Output 2: {output2}
            # Output 3: {output3}
            # """
            response = model.chat(
                tokenizer,
                None,
                question,
                generation_config,
                num_patches_list=None,
                history=None,
            )

            if "\n" in response:
                response = response.replace("\n", " ")

            fw.write(f"{key} | {response}\n")
            print(f"Processed: {key}, Response: {response}")


    