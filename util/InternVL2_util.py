import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import math
import base64
import requests
from io import BytesIO
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def resize_image_with_threshold(image, threshold):
    
    # Get the original dimensions
    original_width, original_height = image.size
    
    # Determine the larger dimension
    larger_dim = max(original_width, original_height)
    
    if larger_dim > threshold:
        # Calculate the scaling factor
        scale_factor = threshold / float(larger_dim)
        
        # Compute the new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize the image
        resized_image = image.resize((new_width, new_height))
        print(f"Image resized to: {new_width}x{new_height}")
    else:
        resized_image = image
        print("Image does not exceed the threshold, no resizing needed.")
    
    return resized_image


def load_image(image_file, input_size=448, max_num=20):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # image = resize_image_with_threshold(image, 1024)
    # print(image.size)
        
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
def ask(img_list, text_prompt):
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
    pixel_values = torch.cat(pixel_value_list, dim=0)
    num_patches_list = [pixel_values.size(0) for pixel_values in pixel_value_list]

    prompt_img_prefix = '\n'.join([f'Image-{x}: <image>' for x in range(1, len(img_list)+1)])
    
    question = prompt_img_prefix + f'\n{text_prompt}'
    with torch.no_grad():
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)
    print(f'text_prompt: {text_prompt}\nAssistant: {response}')
    return response

def ask_cat(img_list, text_prompt):
    
    # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
    pixel_values = torch.cat(pixel_value_list, dim=0)
    
    question = f'<image>\n{text_prompt}'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                   history=None, return_history=True)
    print(f'text_prompt: {text_prompt}\nAssistant: {response}')
    return response


import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def extract_number(filename):
    filename = filename.replace('-1024.jpg', '')
    x = filename.split('-')[-1]
    return int(x)



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map