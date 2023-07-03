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
nerf.load_state_dict(torch.load("torch/torch_NeRF_10000.pth"))

trainImages = "finalDataset/train/images/"
testImages = "finalDataset/test/images/"
testPoses = json.load(open("finalDataset/test/test.json"))['frames']
trainPoses = json.load(open("finalDataset/train/transforms.json"))['frames']

imageList = os.listdir(trainImages)
for i in range(len(trainPoses[1:10])):
    name = trainPoses[i]["file_path"][9:]
    pose = trainPoses[i]["transform_matrix"]
    print(name)
    print(pose)
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
    
    file = "torch/testTrain/{}.png".format(name)
    plt.imsave(file, render[0].detach().cpu().numpy())
