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

images_folder = "finalDataset/train/images"
poses_file = json.load(open("finalDataset/train/transforms.json"))

print("Loading data...")

images = []
poses = []

imageList = os.listdir(images_folder)
for imageFile in imageList:
    image = cv2.imread(images_folder + "/" + imageFile)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    images.append(image)
    poses.append(poses_file["frames"][imageList.index(imageFile)]["transform_matrix"])

images = np.array(images)
poses = np.array(poses)

H, W = images.shape[1:3]
focal_length = 50

print("Datset loaded with:")
print("images {}".format(images.shape))
print("poses {}".format(poses.shape))

images = torch.FloatTensor(images).cuda()
poses = torch.FloatTensor(poses).cuda()

dataset = PixelRayDataset(images, poses, focal_length)
data_loader = data.DataLoader(dataset, batch_size=8000, shuffle=True)

nerf = NeRF(normalize_position=6.0).cuda()
nerf_optimizer = optim.Adam(nerf.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nerf.to(device)
print("\nModel loaded to {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

print("Starting training...\n")
psnrs = []
iternums = []

iteration = -1
maxIterations = 10000

while iteration < maxIterations:
    for _, batch in enumerate(data_loader):
        if iteration < maxIterations:
            iteration += 1
            
            pixels = nerf.render_rays(
                batch['rays_o'],
                batch['rays_d'],  
                2.0, 
                6.0, 
                64, 
                randomly_sample=True, 
                density_noise_std=1.0)

            nerf_optimizer.zero_grad()

            ((pixels - batch['pixels']) ** 2).mean().backward()

            nerf_optimizer.step()

            if iteration % 500 == 0:
                print("Iteration {iteration}/{maxIterations}".format(iteration = iteration, maxIterations = maxIterations))
        else:
            break

torch.save(nerf.state_dict(), "torch/complete_torch_NeRF_{}.pth".format(iteration))
print("\nModel saved at torch_NeRF_{}.pth".format(iteration))