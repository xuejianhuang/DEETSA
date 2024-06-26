import torch
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
import json
import os,random
from collate_fn import *
from util import *
from models import *
import config

def get_transform_Compose():
    return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def split_dataset(dataset_path, train_ratio=config.train_ratio, val_ratio=config.val_ratio):
    with open(dataset_path, encoding='utf-8') as f:
        all_items_dict = json.load(f)

    items = list(all_items_dict.items())
    random.shuffle(items)

    total_items = len(items)
    train_size = int(train_ratio * total_items)
    val_size = int(val_ratio * total_items)

    train_items = dict(items[:train_size])
    val_items = dict(items[train_size:train_size + val_size])
    test_items = dict(items[train_size + val_size:])

    return train_items, val_items, test_items

class DEETSA_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        img_to_text=self.context_data_items_dict[key]['img-to-text']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption,img_to_text

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        direct_path_item = os.path.join(self.data_root_dir, item['direct_path'])
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        captions = load_captions(inv_ann_dict)
        captions += load_captions_weibo(direct_dict)
        captions = captions[:config.max_captions_num]
        imgs = load_imgs_direct_search(self.transform, direct_path_item, direct_dict)
        qImg, qCap,img_to_text = self.load_data(key)
        sample = {'label': label, 'caption': captions, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap,'img_to_text':img_to_text}

        return sample, len(captions), imgs.shape[0],key

    def __len__(self):
        return len(self.context_data_items_dict)

class TETSA_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        img_to_text=self.context_data_items_dict[key]['img-to-text']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption,img_to_text

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        direct_path_item = os.path.join(self.data_root_dir, item['direct_path'])
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        captions = load_captions(inv_ann_dict)
        captions += load_captions_weibo(direct_dict)
        captions = captions[:config.max_captions_num]
        qImg, qCap,img_to_text = self.load_data(key)
        sample = {'label': label, 'caption': captions, 'qImg': qImg, 'qCap': qCap,'img_to_text':img_to_text}

        return sample, len(captions),key

    def __len__(self):
        return len(self.context_data_items_dict)

class IETSA_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        img_to_text=self.context_data_items_dict[key]['img-to-text']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption,img_to_text

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        direct_path_item = os.path.join(self.data_root_dir, item['direct_path'])
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        imgs = load_imgs_direct_search(self.transform, direct_path_item, direct_dict)
        qImg, qCap,img_to_text = self.load_data(key)
        sample = {'label': label, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap,'img_to_text':img_to_text}

        return sample, imgs.shape[0],key

    def __len__(self):
        return len(self.context_data_items_dict)

class DEE_Dataset(Dataset):
    def __init__(self, data_items, data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def load_data(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        direct_path_item = os.path.join(self.data_root_dir, item['direct_path'])
        inverse_path_item = os.path.join(self.data_root_dir, item['inv_path'])
        inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding='utf-8'))
        direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding='utf-8'))
        captions = load_captions(inv_ann_dict)
        captions += load_captions_weibo(direct_dict)
        captions = captions[:config.max_captions_num]
        imgs = load_imgs_direct_search(self.transform, direct_path_item, direct_dict)
        qImg, qCap = self.load_data(key)
        sample = {'label': label, 'caption': captions, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap}

        return sample, len(captions), imgs.shape[0],key

    def __len__(self):
        return len(self.context_data_items_dict)

class TSA_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        qCap = self.context_data_items_dict[key]['caption']
        img_to_text = self.context_data_items_dict[key]['img-to-text']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return label, transform_img,qCap,img_to_text,key

    def __len__(self):
        return len(self.context_data_items_dict)

class TI_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        qCap = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)

        return label, qCap, transform_img,key

    def __len__(self):
        return len(self.context_data_items_dict)


class T_Dataset(Dataset):
    def __init__(self, data_items):
        self.context_data_items_dict = data_items
        self.idx_to_keys = list(self.context_data_items_dict.keys())

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        qCap = self.context_data_items_dict[key]['caption']
        return label, qCap,key

    def __len__(self):
        return len(self.context_data_items_dict)


class I_Dataset(Dataset):
    def __init__(self, data_items,data_root_dir):
        self.context_data_items_dict = data_items
        self.data_root_dir = data_root_dir
        self.idx_to_keys = list(self.context_data_items_dict.keys())
        self.transform = get_transform_Compose()

    def __getitem__(self, idx):
        key = self.idx_to_keys[idx]
        item = self.context_data_items_dict.get(key)
        label = torch.tensor(int(item['label']))
        image_path = os.path.join(self.data_root_dir, self.context_data_items_dict[key]['image_path'])
        pil_img = load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return label.to(config.device), transform_img.to(config.device),key

    def __len__(self):
        return len(self.context_data_items_dict)


def getModelAndData(model_name='completed', dataset='weibo'):

    if dataset =='weibo':
        data_root_dir = config.weibo_dataset_dir
    else:
        data_root_dir=config.twitter_dataset_dir

    dataset_path=os.path.join(data_root_dir, 'dataset_items_merged.json')

    train_items, val_items, test_items = split_dataset(dataset_path)

    # DEETSA/TETSA/IETSA/DEE/TSA/TIM/text/image/
    if model_name== 'DEETSA':
        train_dataset = DEETSA_Dataset(train_items, data_root_dir)
        val_dataset = DEETSA_Dataset(val_items, data_root_dir)
        test_dataset = DEETSA_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      collate_fn=collate_DEETSA)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=collate_DEETSA)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_DEETSA)
        model = DEETSA().to(config.device)

    elif model_name == 'TETSA':
        train_dataset = TETSA_Dataset(train_items, data_root_dir)
        val_dataset = TETSA_Dataset(val_items, data_root_dir)
        test_dataset = TETSA_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      collate_fn=collate_TETSA)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=collate_TETSA)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_TETSA)
        model = TETSA().to(config.device)
    elif model_name == 'IETSA':
        train_dataset = IETSA_Dataset(train_items, data_root_dir)
        val_dataset = IETSA_Dataset(val_items, data_root_dir)
        test_dataset = IETSA_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      collate_fn=collate_IETSA)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=collate_IETSA)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_IETSA)
        model = IETSA().to(config.device)

    elif model_name=='DEE':
        train_dataset = DEE_Dataset(train_items, data_root_dir)
        val_dataset = DEE_Dataset(val_items, data_root_dir)
        test_dataset = DEE_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      collate_fn=collate_DEE)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=collate_DEE)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_DEE)
        model=DEE().to(config.device)


    elif model_name ==  'TSA':
        train_dataset = TSA_Dataset(train_items, data_root_dir)
        val_dataset = TSA_Dataset(val_items, data_root_dir)
        test_dataset = TSA_Dataset(test_items, data_root_dir)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                      collate_fn=collate_TSA)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                                    collate_fn=collate_TSA)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_TSA)
        model = TSA().to(config.device)


    elif model_name=='TIM':
        train_dataset = TI_Dataset(train_items, data_root_dir)
        val_dataset = TI_Dataset(val_items, data_root_dir)
        test_dataset=TI_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_TI)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_TI)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_TI)
        model = TIM().to(config.device)

    elif model_name=='text':
        train_dataset = T_Dataset(train_items)
        val_dataset = T_Dataset(val_items)
        test_dataset=T_Dataset(test_items)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,collate_fn=collate_text)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,collate_fn=collate_text)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,collate_fn=collate_text)
        model=TM().to(config.device)
    elif model_name=='image':
        train_dataset = I_Dataset(train_items, data_root_dir)
        val_dataset = I_Dataset(val_items, data_root_dir)
        test_dataset=I_Dataset(test_items, data_root_dir)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        model = IM().to(config.device)

    return model, train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    dataset_path = os.path.join(config.twitter_dataset_dir, 'dataset_items_merged.json')
    train,val,test=split_dataset(dataset_path)
    print(len(train),len(val),len(test))