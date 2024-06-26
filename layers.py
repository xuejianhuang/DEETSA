import os
import torch
import torch.nn as nn
import torchvision.models as models
import config


class EncoderCNN(nn.Module):
    def __init__(self, resnet_arch='resnet101', pretrained_path=None):
        """
        Initialize the EncoderCNN class.

        Parameters:
        resnet_arch (str): The architecture of the ResNet model to use. Default is 'resnet101'.
        pretrained_path (str): Path to the pretrained model weights. If None, use the default pretrained weights from torchvision.

        Returns:
        None
        """
        super(EncoderCNN, self).__init__()

        # Dictionary to map architecture names to torchvision models
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }

        # Check if the provided architecture is valid
        if resnet_arch not in resnet_models:
            raise ValueError(
                f"Unsupported resnet architecture: {resnet_arch}. Supported architectures: {list(resnet_models.keys())}")

        # Load the appropriate ResNet model
        if pretrained_path is None:
            resnet = resnet_models[resnet_arch](weights='IMAGENET1K_V2')
        else:
            resnet = resnet_models[resnet_arch](weights=None)
            resnet.load_state_dict(torch.load(pretrained_path))

        # Remove the fully connected layer and adaptive pooling layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, features='pool'):
        """
        Forward pass of the EncoderCNN.

        Parameters:
        images (Tensor): Batch of images.
        features (str): Specifies the type of features to extract. Default is 'pool'.

        Returns:
        Tensor: Extracted features.
        """
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = out.view(out.size(0), -1)
        return out


if __name__ == '__main__':
    # Example usage
    pretrained_path = config.resnet_path  # Path to local model weights file
    encoder_cnn = EncoderCNN(resnet_arch='resnet101', pretrained_path=pretrained_path)

    # Initialize without using local pretrained weights
    encoder_cnn_no_pretrain = EncoderCNN(resnet_arch='resnet50')
