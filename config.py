import torch
from transformers import BertTokenizer, BertModel

# Directories for datasets and model saving
weibo_dataset_dir='./data/Weibo/'
twitter_dataset_dir='./data/Twitter/'
model_saved_path='./best_model/'

# BERT model and tokenizer directories
bert_dir='./bert-base-multilingual-uncased/'
_tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-uncased')
_bert_model= BertModel.from_pretrained("./bert-base-multilingual-uncased")
bert_freeze = False  # Set to True to freeze BERT layers during training


# Paths to ResNet model weights
resnet101_path='./ResNet/resnet101-5d3b4d8f.pth'
resnet50_path='./ResNet/resnet50-11ad3fa6.pth'

# Device configuration for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of output classes for the classifier
num_classes=3

# Text and image processing parameters
text_max_length = 40  # Maximum length of text sequences
max_images_num = 5  # Maximum number of visual evidence
max_captions_num = 5  # Maximum number of textual evidence
text_dim = 768  # Dimension of text embeddings
img_dim = 2048  # Dimension of image embeddings
hidden_dim = 768  # Hidden dimension size
classifier_hidden_dim = 128  # Hidden dimension size for the classifier


# Attention mechanism parameters
att_num_heads = 8  # Number of attention heads
att_dropout = 0  # Dropout rate for attention layers
f_dropout = 0  # Dropout rate for fully connected layers


# Training parameters
batch_size = 32  # Batch size for training
epoch = 8  # Number of training epochs
patience = 2  # Patience for early stopping
lr = 5e-5  # Learning rate
decayRate = 0.96  # Learning rate decay rate

# Data split ratios
train_ratio = 0.8  # Proportion of data for training
val_ratio = 0.1  # Proportion of data for validation
test_ratio = 0.1  # Proportion of data for testing
