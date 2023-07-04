from nerf.model import NeRF
from nerf.dataset import PixelRayDataset
import matplotlib.pyplot as plt
import numpy as np
import io, json, cv2, io, os
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd

nerf = NeRF(normalize_position=6.0).cuda()
nerf.load_state_dict(torch.load("torch/complete_torch_NeRF_10000.pth"))

trainImages = "completeDataset/images/"
videoImages = "torch/rotation/images/"
testImages = "finalDataset/test/images/"
testPoses = json.load(open("finalDataset/test/test.json"))['frames']
trainPoses = json.load(open("completeDataset/transforms.json"))['frames']
videoPoses = json.load(open("torch/rotation/video.json"))['frames']

imageList = os.listdir(videoImages)
for i in range(len(videoPoses)):
    name = videoPoses[i]["file_path"][9:]
    pose = videoPoses[i]["transform_matrix"]
    pose = torch.FloatTensor([pose]).cuda()
    with torch.no_grad():    
        render = nerf.render_image(
            pose[..., :3,  3], 
            pose[..., :3, :3],  
            256, 
            256, 
            50, 
            2.0, 
            6.0, 
            64)
    
    file = "torch/rotation/renders/{}.png".format(name)
    plt.imsave(file, render[0].detach().cpu().numpy())
