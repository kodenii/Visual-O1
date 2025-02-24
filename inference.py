# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
import io
# import gradio as gr
import torch
import argparse
import json
import os
from PIL import Image
import cv2
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch


# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


from task_adapter.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np

from gpt4v import Agent
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

from file_utils import read_json, read_jsonl
import matplotlib.colors as mcolors
import tqdm
import re
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

gpt_config = read_json("gpt_config.json")
MODEL_NAME = gpt_config["MODEL_NAME"]
API_KEY = gpt_config["API_KEY"]
API_VERSION = gpt_config["API_VERSION"]
AZURE_ENDPOINT = gpt_config["AZURE_ENDPOINT"]


'''
build args
'''
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./checkpoints/swinl_only_sam_many2many.pth"
sam_ckpt = "./checkpoints/sam_vit_h_4b8939.pth"
seem_ckpt = "./checkpoints/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)


'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

history_images = []
history_masks = []
history_texts = []
@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:                
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(image.convert('L')))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, image, text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, image, spatial_masks, text_size, label_mode, alpha, anno_mode)

        return Image.fromarray(output), mask

def extract_numbers_from_last_answer(text):
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answers = answer_pattern.findall(text)
    
    if not answers:
        number_pattern = re.compile(r'\[([\d,\s]+)\]')
        matches = number_pattern.findall(text)
        if not matches:
            number_pattern = re.compile(r'\d+')
            matches = number_pattern.findall(text)
            if not matches:
                return []
            else:
                return [int(num) for num in matches]
        numbers = []
        for match in matches:
            numbers.extend([int(num.strip()) for num in match.split(',') if num.strip()])
        return numbers
    last_answer = answers[-1]
    number_pattern = re.compile(r'\[(\d+)\]')
    numbers = number_pattern.findall(last_answer)
    return_list = [int(num) for num in numbers]
    if len(return_list) == 0:
        number_pattern = re.compile(r'\d+')
        numbers = number_pattern.findall(last_answer)
        return_list = [int(num) for num in numbers]
    return return_list



def highlight(masks, res):
    res = extract_numbers_from_last_answer(res)
    all_mask = np.zeros(masks[0]['segmentation'].shape)
    for i, r in enumerate(res):
        length = len(masks)
        if int(r)-1 >= length:
            continue
        mask_i = masks[int(r)-1]['segmentation']
        all_mask = np.logical_or(all_mask, mask_i)
    return all_mask


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    else:
        return intersection / union, intersection, union



def do_ris(raw_image, text, instruction, output, id, infer_mode, name, slider=1.0, mode='Automatic', slider_alpha=0.05, label_mode='Number', anno_mode='Mark'):
    anno_image, masks = inference(raw_image, slider, mode, slider_alpha, label_mode, anno_mode)
    agent = Agent(MODEL_NAME, API_KEY, API_VERSION, AZURE_ENDPOINT, instruction)
    seg_res = agent.chat(text=text, image=anno_image)
    if not os.path.exists(f"./{output}/{infer_mode}/{name}"):
        os.makedirs(f"./{output}/{infer_mode}/{name}")
    with open(f"./{output}/{infer_mode}/{name}/gpt_response.jsonl", 'a') as f:
        data = {'id': id, 'text': text, 'response': seg_res}
        f.write(json.dumps(data) + '\n')
    mask = highlight(masks, seg_res)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(raw_image.size))
    return mask


def run(file_name, mode):
    print("inference")
    name = file_name.split(".")[0]
    output = "output"
    prompts = read_json("instructions/prompt.json")
    if not os.path.exists(f"./{output}/{mode}"):
        os.makedirs(f"./{output}/{mode}")
    if mode == "empirical":
        json_dir = f"data/deblur_{file_name}"
        instruction = prompts["default_infer_instr"]
    else:
        json_dir = f"data/{file_name}"
        instruction = prompts["instantial_infer_instr"]
        instruction.format(budget=10)
    img_base_dir = "./images/"
    dataset = read_jsonl(json_dir)
    ious = []
    intersection_sum = 0
    union_sum = 0
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            data = dataset[i]
            img_name = data['img_name']
            img_dir = os.path.join(img_base_dir, img_name)
            if mode == "empirical":
                origin_text = data['sent']
                deblur_text = data['disambiguous_sent']
                deblur_text = deblur_text.lower()
                input_text = "I have labeled a bright numeric ID at the center for each visual object in the image. Please tell me the IDs for:" + deblur_text
                
            else:
                text = data['sent']
                input_text = "Description: " + text + "\nProvide a detailed, step-by-step solution to a given question."
            ground_seg = data["segmentation"][0]
            polys = []
            poly = np.array(ground_seg, dtype=np.int32).reshape((int(len(ground_seg) / 2), 2))
            polys.append(poly)
            raw_image = Image.open(img_dir).convert('RGB')
            width, height = raw_image.size
            mask = np.zeros((height, width, 1))
            ground_mask = cv2.fillPoly(mask, polys, 1)
            ground_mask = ground_mask.squeeze()
            sent_id = data['sent_id']
            predict_mask = do_ris(raw_image, input_text, instruction, output, sent_id, mode, name)
            
            if not os.path.exists(f"./{output}/{mode}/{name}/mask_np"):
                os.makedirs(f"./{output}/{mode}/{name}/mask_np")
            if not os.path.exists(f"./{output}/{mode}/{name}/mask_image"):
                os.makedirs(f"./{output}/{mode}/{name}/mask_image")
            np.save(f"./{output}/{mode}/{name}/mask_np/mask_sent_id_{sent_id}", predict_mask)
            predict_mask1 = predict_mask[:,:,0]
            iou, intersection, union = calculate_iou(ground_mask, predict_mask1)
            ious.append(iou)
            intersection_sum += intersection
            union_sum += union
            new_data = {'sent_id': data['sent_id'], 'img_name': data['img_name'], 'sent': input_text, "iou":float(iou),"intersection":int(intersection), "union": int(union),"segmentation":data['segmentation']}
            with open(f"./{output}/{mode}/{name}/iou_result.jsonl", 'a') as f:
                json.dump(new_data, f)
                f.write("\n")

            masked = np.array(raw_image) * predict_mask
            masked = Image.fromarray(masked.astype(np.uint8))
            masked.save(f"./{output}/{mode}/{name}/mask_image/masked_sent_id_{sent_id}.png")
            pbar.update(1)
    
    gious = sum(ious) / len(ious)
    cious = intersection_sum / union_sum
    print("ciou:", cious)
    print("giou:", gious)
    print("len:", len(ious))
    
