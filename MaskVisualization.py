import os
import torch
from matplotlib import pyplot as plt

masks = ['dog']
for mask in masks:
    masks_collection = torch.load('masks/{}.pth'.format(mask))
    # print(masks_collection.keys())
    for fr_idx in masks_collection.keys():
        frame = masks_collection[fr_idx][0][0]
        frame = frame.cpu().detach().numpy()
        for layer_idx, layer in enumerate(frame):
            layer = layer[0]
            save_dir = 'masks/{}_split/{}/'.format(mask, layer_idx)
            os.makedirs(save_dir, exist_ok=True)
            plt.imsave('masks/{}_split/{}/{}.png'.format(mask, layer_idx, fr_idx), layer)
