import torch
import config

def collate_DEETSA(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    samples_key=[item[3] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    img_to_text_batch=[]
    img_batch = []
    cap_batch = []
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(max_captions_len - cap_len):
            captions.append("")
        captions_tokenizer = config._tokenizer(captions, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                               truncation=True).to(config.device)
        cap_batch.append(captions_tokenizer)  #cap_batch=[batch_size,max_captions_len,tokenizer]

        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len - sample['imgs'].shape[0], sample['imgs'].shape[1], sample['imgs'].shape[2], sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len - sample['imgs'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['imgs'], torch.zeros(padding_size)), dim=0)
        img_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])  # [3, 224, 224]
        qCap_batch.append(sample['qCap'])  #qCap_batch 数组 length=batch_size
        img_to_text_batch.append(sample['img_to_text'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device) #qCap_batch=[batch_size,tokenizer]
    img_to_text_batch=config._tokenizer(img_to_text_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    img_batch = torch.stack(img_batch, dim=0).to(config.device)  #img_batch.shape=[batch_size,max_images_len,3,224,224]
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)   #qImg_batch.shape=[batch_size,3,224,224]
    labels = torch.stack(labels, dim=0).to(config.device)  #labels=[batch_size]
    return labels, cap_batch, img_batch, qCap_batch, qImg_batch,img_to_text_batch,samples_key

def collate_TETSA(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    samples_key = [item[2] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    img_to_text_batch=[]
    cap_batch = []
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(max_captions_len - cap_len):
            captions.append("")
        captions_tokenizer = config._tokenizer(captions, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                               truncation=True).to(config.device)
        cap_batch.append(captions_tokenizer)  #cap_batch=[batch_size,max_captions_len,tokenizer]



        qImg_batch.append(sample['qImg'])  # [3, 224, 224]
        qCap_batch.append(sample['qCap'])  #qCap_batch 数组 length=batch_size
        img_to_text_batch.append(sample['img_to_text'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device) #qCap_batch=[batch_size,tokenizer]
    img_to_text_batch=config._tokenizer(img_to_text_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)   #qImg_batch.shape=[batch_size,3,224,224]
    labels = torch.stack(labels, dim=0).to(config.device)  #labels=[batch_size]
    return labels, cap_batch, qCap_batch, qImg_batch,img_to_text_batch,samples_key

def collate_IETSA(batch):
    samples = [item[0] for item in batch]
    max_images_len = max([item[1] for item in batch])
    samples_key = [item[2] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    img_to_text_batch=[]
    img_batch = []
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])

        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len - sample['imgs'].shape[0], sample['imgs'].shape[1], sample['imgs'].shape[2], sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len - sample['imgs'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['imgs'], torch.zeros(padding_size)), dim=0)
        img_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])  # [3, 224, 224]
        qCap_batch.append(sample['qCap'])  #qCap_batch 数组 length=batch_size
        img_to_text_batch.append(sample['img_to_text'])

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device) #qCap_batch=[batch_size,tokenizer]
    img_to_text_batch=config._tokenizer(img_to_text_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    img_batch = torch.stack(img_batch, dim=0).to(config.device)  #img_batch.shape=[batch_size,max_images_len,3,224,224]
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)   #qImg_batch.shape=[batch_size,3,224,224]
    labels = torch.stack(labels, dim=0).to(config.device)  #labels=[batch_size]
    return labels, img_batch, qCap_batch, qImg_batch,img_to_text_batch,samples_key

def collate_DEE(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    samples_key = [item[3] for item in batch]
    qImg_batch = []
    qCap_batch=[]
    img_batch = []
    cap_batch = []
    labels = []
    for j in range(len(samples)):
        sample = samples[j]
        labels.append(sample['label'])
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(max_captions_len - cap_len):
            captions.append("")
        captions_tokenizer = config._tokenizer(captions, return_tensors='pt', max_length=config.text_max_length, padding='max_length',
                                               truncation=True).to(config.device)
        cap_batch.append(captions_tokenizer)  #cap_batch=[batch_size,max_captions_len,tokenizer]

        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len - sample['imgs'].shape[0], sample['imgs'].shape[1], sample['imgs'].shape[2], sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len - sample['imgs'].shape[0], 3,224,224)
        padded_mem_img = torch.cat((sample['imgs'], torch.zeros(padding_size)), dim=0)
        img_batch.append(padded_mem_img)

        qImg_batch.append(sample['qImg'])  # [3, 224, 224]
        qCap_batch.append(sample['qCap'])  #qCap_batch 数组 length=batch_size

    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device) #qCap_batch=[batch_size,tokenizer]
    img_batch = torch.stack(img_batch, dim=0).to(config.device)  #img_batch.shape=[batch_size,max_images_len,3,224,224]
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)   #qImg_batch.shape=[batch_size,3,224,224]
    labels = torch.stack(labels, dim=0).to(config.device)  #labels=[batch_size]
    return labels, cap_batch, img_batch, qCap_batch, qImg_batch,samples_key

def collate_TSA(batch):
    labels=[]
    qImg_batch = []
    qCap_batch=[]
    img_to_text_batch = []
    samples_key=[]
    for item in batch:
        labels.append(item[0])
        qImg_batch.append(item[1])
        qCap_batch.append(item[2])
        img_to_text_batch.append(item[3])
        samples_key.append(item[4])
    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)  # qCap_batch=[batch_size,tokenizer]
    img_to_text_batch=config._tokenizer(img_to_text_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    qImg_batch = torch.stack(qImg_batch, dim=0).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)  # labels=[batch_size]
    return labels,qImg_batch,qCap_batch,img_to_text_batch,samples_key

def collate_TI(batch):
    labels=[]
    qCap_batch=[]
    qImg_batch=[]
    samples_key = []
    for item in batch:
        labels.append(item[0])
        qCap_batch.append(item[1])
        qImg_batch.append(item[2])
        samples_key.append(item[3])
    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(config.device)
    labels = torch.stack(labels, dim=0).to(config.device)
    qImg_batch=torch.stack(qImg_batch,dim=0).to(config.device)
    return labels,qCap_batch,qImg_batch,samples_key

def collate_text(batch):
    labels=[]
    qCap_batch=[]
    samples_key = []
    for item in batch:
        labels.append(item[0])
        qCap_batch.append(item[1])
        samples_key.append(item[2])
    #text_max_length = get_maxlength(qCap_batch)
    qCap_batch = config._tokenizer(qCap_batch, return_tensors='pt', max_length=config.text_max_length,
                                   padding='max_length', truncation=True).to(
        config.device)  # qCap_batch=[batch_size,tokenizer]
    labels = torch.stack(labels, dim=0).to(config.device)  # labels=[batch_size]
    return labels,qCap_batch,samples_key

