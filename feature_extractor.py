import logging
logging.disable(logging.CRITICAL)
import numpy as np, time as timer, torch, torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet34, resnet18, resnet50, resnet101
from PIL import Image
_encoders = {'resnet34':resnet34, 
 'resnet18':resnet18,  'resnet50':resnet50,  'resnet101':resnet101}
_transforms = {'resnet34':transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [
   0.229, 0.224, 0.225])]), 
 'resnet18':transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [
   0.229, 0.224, 0.225])]), 
 'resnet50':transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [
   0.229, 0.224, 0.225])]), 
 'resnet101':transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [
   0.229, 0.224, 0.225])])}

class Encoder(nn.Module):
    def __init__(self, model_type):
        super(Encoder, self).__init__()
        self.model_type = model_type
        self.model = _encoders[model_type](pretrained=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = Identity().to(self.device)

    def forward(self, x):
        x = self.model(x)
        return x

    def get_image_transform(self):
        return _transforms[self.model_type]

    def get_features(self, x):
        z = self.model(x[:, :3, :, :]).squeeze().cpu().detach().numpy()
        if x.shape[1] > 3:
            z = np.hstack((z, self.model(x[:, 3:6, :, :]).squeeze().cpu().detach().numpy()))
        if x.shape[1] > 6:
            z = np.hstack((z, self.model(x[:, 6:9, :, :]).squeeze().cpu().detach().numpy()))
        return z


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
