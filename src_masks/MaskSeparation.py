import os
import torch
import shutil
from matplotlib import pyplot as plt

masks = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees',
                   'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane',
                   'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch',
                   'bike-packing', 'dogs-jump', 'gold-fish', 'india', 'judo', 'lab-coat', 'loading', 'mbike-trick',
                   'pigs', 'shooting']
masks_folder = "masks"
if os.path.exists(masks_folder):
    shutil.rmtree(masks_folder)
os.makedirs(masks_folder)

for mask in masks:
    masks_collection = torch.load('DAVIS_Attn/{}.pth'.format(mask))
    for fr_idx in masks_collection.keys():
        frame = masks_collection[fr_idx][0][0]
        frame = frame.cpu().detach().numpy()
        for layer_idx, layer in enumerate(frame):
            layer = layer[0]
            print("Layer {}:".format(layer_idx))
            save_dir = 'masks/{}_split/{}/'.format(mask, layer_idx)
            os.makedirs(save_dir, exist_ok=True)
            plt.imsave('masks/{}_split/{}/{}.png'.format(mask, layer_idx, fr_idx), layer)
