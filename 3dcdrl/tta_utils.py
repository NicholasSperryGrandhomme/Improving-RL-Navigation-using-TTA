import os
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Resize, Grayscale
import torch.nn.functional as F

from models import Encoder

#File containing all TTA related functions and classes.
#The following were taking from https://github.com/nicklashansen/policy-adaptation-during-deployment:
# tie_weights, rotate_with_single_label, rotate. 
# A lot of the rest is based of the github mentioned above

#Tie the weights of 2 layers so that they are updated together
def tie_weights(src,trg):
    trg.weight = src.weight
    trg.bias = src.bias

#Return correct rotation corresponding to the label
def rotate_with_single_label(x, label):
    if label == 1:
        return x.flip(2).transpose(1, 2)
    elif label == 2:
        return x.flip(2).flip(1)
    elif label == 3:
        return x.transpose(1, 2).flip(2)
    return x
#Randomly rotate images in batch and return corresponding labels
def rotate(x):
    images = []
    labels = torch.randint(4, (x.size(0),), dtype=torch.long).to(x.device)
    for img, label in zip(x, labels):
        img = rotate_with_single_label(img, label)
        images.append(img.unsqueeze(0))
        #print(img.shape)
    return torch.cat(images,dim=0), labels

#Return grayscale image corresponding to the label
def grayscale_with_label(x, label):
    if label == 1:
        return Grayscale(num_output_channels=3)(x)
    return x
#Randomly grayscale images in a batch and return corresponding labels
def grayscale(x):
    images = []
    labels = torch.randint(2, (x.size(0),), dtype=torch.long).to(x.device)
    for img, label in zip(x, labels):
        img = grayscale_with_label(img, label)
        images.append(img.unsqueeze(0))
    return torch.cat(images), labels

#The TTA method's own independent network. This is used to predict
#if an image has been rotated, or if it has been grayscaled.
class TTAFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim, out=4):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim,hidden_dim), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim,hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, out))
    def forward(self, h):
        return self.trunk(h)

#The TTA encapsulated in one class.
class TTAAgent(nn.Module):
    def __init__(self, use_rot=True, obs_shape=(3,64,112), hidden_size=64):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Create Encoder network, which will be shared with the Agent
        self.ss_encoder = nn.Sequential(Encoder(),nn.Flatten(), 
                                        nn.Linear(256, 64), nn.LayerNorm(64))

        #Determine what function to use for the Self-Supervised task
        self.use_rot = use_rot
        if use_rot:
            self.tta_method = rotate
        else:
            self.tta_method = grayscale
           
        #Initialise the linear layers of the TTA function
        self.tta = TTAFunction(64, hidden_size*4).to(device)
        #Initialise the optimisers
        self.init_ss_optimizers()

        #Helper function to resize images
        self.resize = Resize((64,64))
    
    #Function used to initialise the optimisers for the TTA function
    def init_ss_optimizers(self, encoder_lr=1e-3, ss_lr=1e-3):
        if self.ss_encoder is not None:
            self.encoder_optimizer =  torch.optim.Adam(
                self.ss_encoder.parameters(), lr=encoder_lr
            )
        
        self.tta_optimizer =  torch.optim.Adam(
            self.tta.parameters(), lr=ss_lr
        )

    #This function is used to tie the weights of the Agent's
    #Encoder to the TTA's encoder
    def copy_conv_weights(self, source):
        for i, layer in enumerate(source):
            if isinstance(layer, nn.Conv2d):
                tie_weights(layer, self.ss_encoder[0].conv[i])
    
    def update_tta(self, obs, L=None, step=None):
        #Need images to be square for the rotation prediction
        obs = self.resize(obs)

        #Create the self-labelled data
        obs, label = self.tta_method(obs)
        h = self.ss_encoder(obs)

        #Get prediction and step
        pred = self.tta(h)
        tta_loss = F.cross_entropy(pred, label)

        self.encoder_optimizer.zero_grad()
        self.tta_optimizer.zero_grad()
        tta_loss.backward()

        self.encoder_optimizer.step()
        self.tta_optimizer.step()

        #Return loss to measure progress
        return tta_loss.item()

    #Helper function to save TTA function weights
    def save(self):
        if self.use_rot:
            name = 'RotationAdaptation.pth.tar'
        else:
            name = 'GrayscaleAdaptation.pth.tar'
        checkpoint = {'encoder': self.ss_encoder.state_dict(),
                      'tta': self.tta.state_dict(),
                      'tta_optimizer': self.tta_optimizer.state_dict(),
                      'encoder_optimizer': self.encoder_optimizer.state_dict()}
        
        torch.save(checkpoint, 'tta_models/'+name)
    
    #Helper functions to load the TTA function weights
    def load(self):
        if self.use_rot:
            name = 'RotationAdaptation.pth.tar'
            
        else:
            name = 'GrayscaleAdaptation.pth.tar'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoints = torch.load('tta_models/'+name, map_location=device)
        #TODO: small fix. Made small mistake and saved as 'rotation'.
        if 'rotation' in checkpoints:
            checkpoints['tta'] = checkpoints.pop('rotation')
            checkpoints['tta_optimizer'] = checkpoints.pop('rotation_optimizer')
        self.ss_encoder.load_state_dict(checkpoints['encoder'])
        self.tta.load_state_dict(checkpoints['tta'])
        self.tta_optimizer.load_state_dict(checkpoints['tta_optimizer'])
        self.encoder_optimizer.load_state_dict(checkpoints['encoder_optimizer'])
        

    
        
