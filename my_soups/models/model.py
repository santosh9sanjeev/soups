import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CombinedModel, self).__init__()
        
        self.feature_extractor = base_model
        self.classifier = nn.Linear(512,num_classes)
    

    def forward(self, x):
        print(x.shape)
        # Forward pass through the ResNet-based feature extractor
        x = self.feature_extractor(x)
        print('after feature extraction', x.shape)
        # Global average pooling (GAP)
        # x = torch.mean(x, dim=(2, 3))  # Assuming the output shape is (batch_size, 512, H, W)

        # Forward pass through the linear classification layer
        x = self.classifier(x)
        print('after class', x.shape)
        return x