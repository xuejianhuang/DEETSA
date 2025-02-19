import numpy as np
import torch
import os,random
from PIL import Image
import config
import argparse
from models import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='MRD')
    parser.add_argument('--dataset', type=str, default='twitter')
    parser.add_argument('--model', type=str, default='DEETSA')  #DEETSA/TETSA/DEE/TSA/TIM/I_TSA/T_TSA/text/image/
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2024, help="The random seed for initialization.")
    args = parser.parse_args()
    return args

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>', '')
    input_str = input_str.replace('</b>', '')
    return input_str

def load_img_pil(image_path):
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            # 将图片转换为RGB格式，无论原始格式如何
            img = img.convert('RGB')
            return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_captions(inv_dict):
    captions = []
    pages_with_captions_keys = ['all_fully_matched_captions', 'all_partially_matched_captions']
    for key1 in pages_with_captions_keys:
        if key1 in inv_dict.keys():
            for page in inv_dict[key1]:
                if 'title' in page.keys():
                    item = page['title']
                    item = process_string(item)
                    captions.append(item)
                if 'caption' in page.keys():
                    sub_captions_list = []
                    unfiltered_captions = []
                    for key2 in page['caption']:
                        sub_caption = page['caption'][key2]
                        sub_caption_filter = process_string(sub_caption)
                        if sub_caption in unfiltered_captions: continue
                        sub_captions_list.append(sub_caption_filter)
                        unfiltered_captions.append(sub_caption)
                    captions = captions + sub_captions_list

    pages_with_title_only_keys = ['partially_matched_no_text', 'fully_matched_no_text']
    for key1 in pages_with_title_only_keys:
        if key1 in inv_dict.keys():
            for page in inv_dict[key1]:
                if 'title' in page.keys():
                    title = process_string(page['title'])
                    captions.append(title)
    return captions

def load_captions_weibo(direct_dict):
    captions = []
    keys = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
    for key1 in keys:
        if key1 in direct_dict.keys():
            for page in direct_dict[key1]:
                if 'page_title' in page.keys():
                    item = page['page_title']
                    item = process_string(item)
                    captions.append(item)
                if 'caption' in page.keys():
                    sub_captions_list = []
                    unfiltered_captions = []
                    for key2 in page['caption']:
                        sub_caption = page['caption'][key2]
                        sub_caption_filter = process_string(sub_caption)
                        if sub_caption in unfiltered_captions: continue
                        sub_captions_list.append(sub_caption_filter)
                        unfiltered_captions.append(sub_caption)
                    captions = captions + sub_captions_list
    return captions

def load_imgs_direct_search(transform,item_folder_path, direct_dict):
    list_imgs_tensors = []
    keys_to_check = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
    for key1 in keys_to_check:
        if key1 in direct_dict.keys():
            for page in direct_dict[key1]:
                image_path = os.path.join(item_folder_path, page['image_path'].split('/')[-1])
                if os.path.exists(image_path):
                    try:
                        pil_img = load_img_pil(image_path)
                        transform_img = transform(pil_img)
                        list_imgs_tensors.append(transform_img)
                    except Exception as e:
                        print(image_path,e)
                else:
                    print(" No such file:",image_path)
                if len(list_imgs_tensors)>=config.max_images_num:
                    list_imgs_tensors = torch.stack(list_imgs_tensors, axis=0)
                    return list_imgs_tensors
    if len(list_imgs_tensors)>0:
        list_imgs_tensors=torch.stack(list_imgs_tensors, axis=0)
    else:
        list_imgs_tensors=torch.tensor(list_imgs_tensors)
    return list_imgs_tensors

def get_maxlength(train_tokens):
    num_tokens = [len (tokens) for tokens in train_tokens]
    num_tokens = np.array (num_tokens)
    max_tokens = np.mean (num_tokens) + 2 * np.std (num_tokens)
    max_tokens = int (max_tokens)
    return max_tokens


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


