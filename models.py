import torch
import torch.nn as nn

from layers import EncoderCNN
import torch.nn.functional as F

import config
#Dual Evidence Enhancement and Text-Image Similarity Awareness
class DEETSA(nn.Module):
    def __init__(self):
        super(DEETSA, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.bert=config._bert_model

        self.c_liner = nn.Linear(config.text_dim*2, config.text_dim)
        self.i_liner = nn.Linear(config.img_dim*2, config.img_dim)
        self.cc_liner=nn.Linear(config.text_dim,config.hidden_dim)
        self.ii_liner=nn.Linear(config.img_dim,config.hidden_dim)
        self.w_ci_liner = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.sim_liner=nn.Linear(config.text_dim,config.hidden_dim)
        self.w_cis_liner = nn.Linear(config.hidden_dim*2, config.hidden_dim)


        self.attention_text = nn.MultiheadAttention(config.text_dim, config.att_num_heads, batch_first=True,dropout=config.att_dropout)
        self.attention_image = nn.MultiheadAttention(config.img_dim, config.att_num_heads, batch_first=True,dropout=config.att_dropout)
        self.attention_sim=nn.MultiheadAttention(config.text_dim, config.att_num_heads, batch_first=True,dropout=config.att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            # nn.Linear(1024, 256),
            # nn.LeakyReLU(),
            # nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        '''
        caps: [batch_size, max_captions_num, tokenizer]
        imgs: [batch_size, max_images_num, 3, 224, 224]
        qCap:  [batch_size,tokenizer]
        qImg: [batch_size,3, 224, 224]
        img_to_text:[batch_size,tokenizer]
        :return:
        '''
        caps, imgs, qCap, qImg, img_to_text=data
        qcap_hidden=self.bert(**qCap)['last_hidden_state']     #[batch,length,768]
        img_to_text_hidden=self.bert(**img_to_text)['last_hidden_state']  #[batch,length,768]
        qcap_feature = qcap_hidden[:, 0, :]  # qcap_feature.shape=[batch_size,768]

        #Text-Image Similarity
        sim_ti,sim_weights=self.attention_sim(qcap_hidden,img_to_text_hidden,img_to_text_hidden) #[batch,length,768]
        sim_ti=torch.mean(sim_ti,dim=1).squeeze(dim=1)

        qImg_feature = self.resnet(qImg)  # qImg_feature.shape=[batch_size,2048]

        caps_features = []
        for cap_tokenizer in caps:
            feature = self.bert(**cap_tokenizer)['last_hidden_state'][:, 0,:]  # CLS tokens   feature.shape=[max_captions_num,768]
            caps_features.append(feature)
        caps_features = torch.stack(caps_features, dim=0)   #caps_feature.shape=[batch_size,max_captions_num,768]
        caps_features,attn_caps_weights = self.attention_text(qcap_feature.unsqueeze(1), caps_features, caps_features) #caps_feature.shape=[batch_size,1,768]

        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #imgs_feature.shape= [max_images_num,2048]
            imgs_features.append(imgs_feature)
        imgs_features = torch.stack(imgs_features, dim=0) #imgs_features.shape=[batch_size,max_images_num,2048]
        imgs_features, attn_imgs_weights = self.attention_image(qImg_feature.unsqueeze(1), imgs_features,imgs_features)  # imgs_features.shape=[batch_size,1,2048]

        #实现qcap_feature和caps_features的自适应融合，得到融合后的 c_feature
        caps_features=caps_features.squeeze(1)
        c_weight= torch.sigmoid(self.c_liner(torch.cat((qcap_feature, caps_features), dim=-1)))
        #c_weight=torch.sigmoid(self.c_liner(caps_features))
        c_feature = c_weight * qcap_feature + (1 - c_weight) * caps_features  #[batch_size,768]

        #c_feature =c_weight * caps_features  # [batch_size,768]

        # 实现qImg_feature和imgs_features的自适应融合，得到融合后的 i_feature
        imgs_features = imgs_features.squeeze(1)
        i_weight = torch.sigmoid(self.i_liner(torch.cat((qImg_feature, imgs_features), dim=-1)))
        #i_weight =torch.sigmoid(self.i_liner( imgs_features))
        i_feature = i_weight * qImg_feature + (1 - i_weight) * imgs_features  # [batch_size,2048]



        #实现c_feature和i_feature的自适应融合，得到融合后的 ci_feature
        cc_feature = self.cc_liner(c_feature)  #[batch_size,768]->[batch_size,768]
        ii_feature=self.ii_liner(i_feature)   #batch_size,2048]->[batch_size,768]
        ci_weight = torch.sigmoid(self.w_ci_liner(torch.cat((cc_feature, ii_feature), dim=-1)))
        ci_feature=ci_weight*cc_feature+(1-ci_weight)*ii_feature  #[batch_size,768]

        #实现ci_feature和sim_ti的自适应融合，得到融合后的 feature
        sim_feature=self.sim_liner(sim_ti)
        ci_sim_weight=torch.sigmoid(self.w_cis_liner(torch.cat((ci_feature, sim_feature),dim=-1)))
        feature=1*ci_feature+(1-ci_sim_weight)*sim_feature

        #feature=torch.cat((qcap_feature,c_feature,qImg_feature,i_feature,sim_ti),dim=-1)

        logits = self.classifier(feature)
        return logits

# Only have Text Evidence,  w/o Image Evidence
class TETSA(nn.Module):
    def __init__(self):
        super(TETSA, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.bert=config._bert_model

        self.c_liner = nn.Linear(config.text_dim * 2, config.text_dim)
        #self.i_liner = nn.Linear(config.img_dim * 2, config.img_dim)
        self.cc_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.ii_liner = nn.Linear(config.img_dim, config.hidden_dim)
        self.w_ci_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.sim_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.w_cis_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.attention_text = nn.MultiheadAttention(config.text_dim, config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.attention_sim=nn.MultiheadAttention(config.text_dim,config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )

        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):

        caps, qCap, qImg, img_to_text=data
        qcap_hidden=self.bert(**qCap)['last_hidden_state']     #[batch,length,768]
        img_to_text_hidden=self.bert(**img_to_text)['last_hidden_state']  #[batch,length,768]
        qcap_feature = qcap_hidden[:, 0, :]  # qcap_feature.shape=[batch_size,768]

        #Text-Image Similarity
        sim_ti,sim_weights=self.attention_sim(qcap_hidden,img_to_text_hidden,img_to_text_hidden) #[batch,length,768]
        sim_ti=torch.mean(sim_ti,dim=1).squeeze(dim=1)

        qImg_feature = self.resnet(qImg)  # qImg_feature.shape=[batch_size,2048]

        caps_features = []
        for cap_tokenizer in caps:
            feature = self.bert(**cap_tokenizer)['last_hidden_state'][:, 0,:]  # CLS tokens   feature.shape=[max_captions_num,768]
            caps_features.append(feature)
        caps_features = torch.stack(caps_features, dim=0)   #caps_feature.shape=[batch_size,max_captions_num,768]
        caps_features,attn_caps_weights = self.attention_text(qcap_feature.unsqueeze(1), caps_features, caps_features) #caps_feature.shape=[batch_size,1,768]


        #实现qcap_feature和caps_features的自适应融合，得到融合后的 c_feature
        caps_features=caps_features.squeeze(1)
        c_weight= torch.sigmoid(self.c_liner(torch.cat((qcap_feature, caps_features), dim=-1)))
        #c_weight=torch.sigmoid(self.c_liner(caps_features))
        c_feature = c_weight * qcap_feature + (1 - c_weight) * caps_features  #[batch_size,768]

       #c_feature =c_weight * caps_features  # [batch_size,768]


        #实现c_feature和qImg_feature的自适应融合，得到融合后的 ci_feature
        cc_feature = self.cc_liner(c_feature)  #[batch_size,768]->[batch_size,768]
        ii_feature=self.ii_liner(qImg_feature)   #batch_size,2048]->[batch_size,768]
        ci_weight = torch.sigmoid(self.w_ci_liner(torch.cat((cc_feature, ii_feature), dim=-1)))
        ci_feature=ci_weight*cc_feature+(1-ci_weight)*ii_feature  #[batch_size,768]

        #实现ci_feature和sim_ti的自适应融合，得到融合后的 feature
        sim_feature=self.sim_liner(sim_ti)  #[batch_size,768]->[batch_size,768]
        ci_sim_weight=torch.sigmoid(self.w_cis_liner(torch.cat((ci_feature, sim_feature),dim=-1)))
        feature=1*ci_feature+(1-ci_sim_weight)*sim_feature

        #feature=torch.cat((qcap_feature,c_feature,qImg_feature,sim_ti),dim=-1)

        logits = self.classifier(feature)
        return logits

# Only have Image Evidence,  w/o Text Evidence
class IETSA(nn.Module):
    def __init__(self):
        super(IETSA, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.bert=config._bert_model


        #self.c_liner = nn.Linear(config.text_dim * 2, config.text_dim)
        self.i_liner = nn.Linear(config.img_dim * 2, config.img_dim)
        self.cc_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.ii_liner = nn.Linear(config.img_dim, config.hidden_dim)
        self.w_ci_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.sim_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.w_cis_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.attention_image = nn.MultiheadAttention(config.img_dim, config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.attention_sim=nn.MultiheadAttention(config.text_dim,config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes)
        )

        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    def forward(self, data):
        imgs, qCap, qImg, img_to_text=data
        qcap_hidden=self.bert(**qCap)['last_hidden_state']     #[batch,length,768]
        img_to_text_hidden=self.bert(**img_to_text)['last_hidden_state']  #[batch,length,768]
        qcap_feature = qcap_hidden[:, 0, :]  # qcap_feature.shape=[batch_size,768]
        #img_to_text_feature=img_to_text_hidden[:,0,:] # [batch_size,768]

        #Text-Image Similarity
        sim_ti,sim_weights=self.attention_sim(qcap_hidden,img_to_text_hidden,img_to_text_hidden) #[batch,length,768]
        sim_ti=torch.mean(sim_ti,dim=1).squeeze(dim=1)

        qImg_feature = self.resnet(qImg)  # qImg_feature.shape=[batch_size,2048]


        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #imgs_feature.shape= [max_images_num,2048]
            imgs_features.append(imgs_feature)
        imgs_features = torch.stack(imgs_features, dim=0) #imgs_features.shape=[batch_size,max_images_num,2048]
        imgs_features, attn_imgs_weights = self.attention_image(qImg_feature.unsqueeze(1), imgs_features,imgs_features)  # imgs_features.shape=[batch_size,1,2048]

        # 实现qImg_feature和imgs_features的自适应融合，得到融合后的 i_feature
        imgs_features = imgs_features.squeeze(1)
        i_weight = torch.sigmoid(self.i_liner(torch.cat((qImg_feature, imgs_features), dim=-1)))
        i_feature = i_weight * qImg_feature + (1 - i_weight) * imgs_features  # [batch_size,2048]

        # 实现qcap_feature和i_feature的自适应融合，得到融合后的 ci_feature
        cc_feature = self.cc_liner(qcap_feature)  #[batch_size,768]->[batch_size,768]
        ii_feature=self.ii_liner(i_feature)   #batch_size,2048]->[batch_size,768]
        ci_weight = torch.sigmoid(self.w_ci_liner(torch.cat((cc_feature, ii_feature), dim=-1)))
        ci_feature=ci_weight*cc_feature+(1-ci_weight)*ii_feature  #[batch_size,768]

        # 实现ci_feature和sim_ti的自适应融合，得到融合后的 feature
        sim_feature=self.sim_liner(sim_ti)  #[batch_size,768]->[batch_size,768]
        ci_sim_weight=torch.sigmoid(self.w_cis_liner(torch.cat((ci_feature, sim_feature),dim=-1)))
        feature=ci_sim_weight*ci_feature+(1-ci_sim_weight)*sim_feature

        logits = self.classifier(feature)
        return logits

# w/o Text-Image Similarity Awareness
class DEE(nn.Module):
    def __init__(self):
        super(DEE, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.bert=config._bert_model


        self.c_liner = nn.Linear(config.text_dim * 2, config.text_dim)
        self.i_liner = nn.Linear(config.img_dim * 2, config.img_dim)
        self.cc_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.ii_liner = nn.Linear(config.img_dim, config.hidden_dim)
        self.w_ci_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.attention_text = nn.MultiheadAttention(config.text_dim, config.att_num_heads, batch_first=True,dropout=config.att_dropout)
        self.attention_image = nn.MultiheadAttention(config.img_dim, config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )

        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        caps, imgs, qCap, qImg=data
        qcap_feature = self.bert(**qCap)['last_hidden_state'][:, 0, :]  # qcap_feature.shape=[batch_size,768]
        qImg_feature = self.resnet(qImg)  # qImg_feature.shape=[batch_size,2048]

        caps_features = []
        for cap_tokenizer in caps:
            feature = self.bert(**cap_tokenizer)['last_hidden_state'][:, 0,:]  # CLS tokens   feature.shape=[max_captions_num,768]
            caps_features.append(feature)
        caps_features = torch.stack(caps_features, dim=0)   #caps_feature.shape=[batch_size,max_captions_num,768]
        caps_features,attn_caps_weights = self.attention_text(qcap_feature.unsqueeze(1), caps_features, caps_features) #caps_feature.shape=[batch_size,1,768]

        imgs_features = []
        for img in imgs:
            imgs_feature = self.resnet(img) #imgs_feature.shape= [max_images_num,2048]
            imgs_features.append(imgs_feature)
        imgs_features = torch.stack(imgs_features, dim=0) #imgs_features.shape=[batch_size,max_images_num,2048]
        imgs_features, attn_imgs_weights = self.attention_image(qImg_feature.unsqueeze(1), imgs_features,imgs_features)  # imgs_features.shape=[batch_size,1,2048]

        #实现qcap_feature和caps_features的自适应融合，得到融合后的 c_feature
        caps_features=caps_features.squeeze(1)
        c_weight= torch.sigmoid(self.c_liner(torch.cat((qcap_feature, caps_features), dim=-1)))
        c_feature = c_weight * qcap_feature + (1 - c_weight) * caps_features  #[batch_size,768]

        # 实现qImg_feature和imgs_features的自适应融合，得到融合后的 i_feature
        imgs_features = imgs_features.squeeze(1)
        i_weight = torch.sigmoid(self.i_liner(torch.cat((qImg_feature, imgs_features), dim=-1)))
        i_feature = i_weight * qImg_feature + (1 - i_weight) * imgs_features  # [batch_size,2048]

        # 实现c_feature和i_feature的自适应融合，得到融合后的 feature
        cc_feature = self.cc_liner(c_feature)  #[batch_size,768]->[batch_size,768]
        ii_feature=self.ii_liner(i_feature)   #batch_size,2048]->[batch_size,768]
        ci_weight = torch.sigmoid(self.w_ci_liner(torch.cat((cc_feature, ii_feature), dim=-1)))
        feature=ci_weight*cc_feature+(1-ci_weight)*ii_feature  #[batch_size,768]


        logits = self.classifier(feature)
        return logits

# w/o Dual Evidence Enhancement
class TSA(nn.Module):
    def __init__(self):
        super(TSA, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.bert=config._bert_model


        # self.c_liner = nn.Linear(config.text_dim * 2, config.text_dim)
        # self.i_liner = nn.Linear(config.img_dim * 2, config.img_dim)
        self.cc_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.ii_liner = nn.Linear(config.img_dim, config.hidden_dim)
        self.w_ci_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.sim_liner = nn.Linear(config.text_dim, config.hidden_dim)
        self.w_cis_liner = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        self.attention_sim=nn.MultiheadAttention(config.text_dim,config.att_num_heads,batch_first=True,dropout=config.att_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )

        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        qImg, qCap, img_to_text=data
        qImg_feature = self.resnet(qImg)  # qImg_feature.shape=[batch_size,2048]

        qcap_hidden=self.bert(**qCap)['last_hidden_state']     #[batch,length,768]
        qcap_feature = qcap_hidden[:, 0, :]  # qcap_feature.shape=[batch_size,768]
        img_to_text_hidden=self.bert(**img_to_text)['last_hidden_state']  #[batch,length,768]
        sim_ti,sim_weights=self.attention_sim(qcap_hidden,img_to_text_hidden,img_to_text_hidden) #[batch,length,768]
        sim_ti=torch.mean(sim_ti,dim=1).squeeze(dim=1)


        # 实现qcap_feature和qImg_feature的自适应融合，得到融合后的 ci_feature
        c_feature = self.cc_liner(qcap_feature)  #[batch_size,768]->[batch_size,768]
        i_feature=self.ii_liner(qImg_feature)   #batch_size,2048]->[batch_size,768]
        ci_weight = torch.sigmoid(self.w_ci_liner(torch.cat((c_feature, i_feature), dim=-1)))
        ci_feature=ci_weight*c_feature+(1-ci_weight)*i_feature  #[batch_size,768]

        # 实现ci_feature和sim_ti的自适应融合，得到融合后的 feature
        sim_feature=self.sim_liner(sim_ti)  #[batch_size,768]->[batch_size,768]
        ci_sim_weight=torch.sigmoid(self.w_cis_liner(torch.cat((ci_feature, sim_feature),dim=-1)))
        feature=ci_sim_weight*ci_feature+(1-ci_sim_weight)*sim_feature

        logits = self.classifier(feature)
        return logits

# Text-Image Model, w/o Both Evidence TSA, Text-Image Similarity Awareness
class TIM(nn.Module):
    def __init__(self):
        super(TIM, self).__init__()
        self.bert = config._bert_model
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.t_linear=nn.Linear(config.text_dim, config.hidden_dim)
        self.i_liner=nn.Linear(config.img_dim, config.hidden_dim)
        self.w_liner=nn.Linear( config.hidden_dim*2, config.hidden_dim)


        self.classifier=nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data):
        qCap, qImg = data
        qcap_feature = self.bert(**qCap)['last_hidden_state'][:, 0, :]  # Shape: [batch_size, 768]
        qImg_feature = self.resnet(qImg)  # Shape: [batch_size,2048]
        qcap_feature=self.t_linear(qcap_feature)
        qImg_feature=self.i_liner(qImg_feature)
        weight=torch.sigmoid(self.w_liner(torch.cat((qcap_feature, qImg_feature),dim=-1)))
        feature=weight*qcap_feature+(1-weight)*qImg_feature

        #feature = torch.cat((qcap_feature, qImg_feature),dim=-1)

        logits = self.classifier(feature)
        return logits

#Text Model, w/o Both Evidence TSA, Text-Image Similarity Awareness, Image
class TM(nn.Module):
    def __init__(self):
        super(TM, self).__init__()
        self.bert=config._bert_model

        self.classifier = nn.Sequential(
            nn.Linear(config.text_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )

        #self.classifier = nn.Linear(768, 3)
        if config.bert_freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    def forward(self, data):
        qCap, = data
        # Extract features from the BERT model. We use the last hidden state of the [CLS] token.
        qcap_feature = self.bert(**qCap)['last_hidden_state'][:, 0, :]  # Shape: [batch_size, 768]
        logits = self.classifier(qcap_feature)
        return logits

#Image Model, w/o Both Evidence TSA, Text-Image Similarity Awareness, Text
class IM(nn.Module):
    def __init__(self):
        super(IM, self).__init__()
        self.resnet = EncoderCNN(resnet_arch='resnet101', pretrained_path=config.resnet101_path)
        self.classifier=nn.Sequential(
            nn.Linear(config.img_dim, config.classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.f_dropout),
            nn.Linear(config.classifier_hidden_dim,  config.num_classes)
        )

    def forward(self, data):
        qImg, = data
        qImg_feature = self.resnet(qImg)
        logits = self.classifier(qImg_feature)
        return logits


if __name__ == '__main__':
    model = DEE("image")
    print(model)
