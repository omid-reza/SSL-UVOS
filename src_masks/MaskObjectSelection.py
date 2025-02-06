import os
import cv2
import shutil
import numpy as np

def get_common_pixels(candidate_path, target_path,target_color=np.array([128, 0, 0]), candidate_color=np.array([68, 1, 84])):
    target_image = cv2.imread(target_path)
    candidate_image = cv2.imread(candidate_path)
    # Resize the candidate image to match the dimensions of the target image
    candidate_image = cv2.resize(candidate_image, (target_image.shape[1], target_image.shape[0]))

    # Convert the candidate image from BGR to RGB format for proper color processing
    candidate_image = cv2.cvtColor(candidate_image, cv2.COLOR_BGR2RGB)
    # Create a binary mask where pixels that do not match the candidate_color are marked as True
    candidate = np.all(candidate_image != candidate_color, axis=-1)

    # Convert the target image from BGR to RGB format for proper color processing
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    # Create a binary mask where pixels that exactly match the target_color are marked as True
    target = np.all(target_image == target_color, axis=-1)
    return np.sum(np.logical_and(candidate, target))


candidate_base = "masks/{}_split/"
new_mask_folder_path = "filtered_masks/"
target_base = "DAVIS/Annotations_unsupervised/480p/{}/00000.png"
parent_folders = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees',
                   'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane',
                   'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch',
                   'bike-packing', 'dogs-jump', 'gold-fish', 'india', 'judo', 'lab-coat', 'loading', 'mbike-trick',
                   'pigs', 'shooting']

if os.path.exists(new_mask_folder_path):
    shutil.rmtree(new_mask_folder_path)
os.makedirs(new_mask_folder_path)

for parent_folder_name in parent_folders:
    t_path = target_base.format(parent_folder_name)
    parent_folder = candidate_base.format(parent_folder_name)
    subfolders = [os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    highest_common_pixel,  highest_common_pixel_subfolder= 0, None
    for subfolder in subfolders:
        c_path = os.path.join(subfolder, "0.png")
        common_pixels = get_common_pixels(candidate_path=c_path, target_path=t_path)
        if common_pixels > highest_common_pixel:
            highest_common_pixel, highest_common_pixel_subfolder = common_pixels, subfolder
    shutil.copytree(highest_common_pixel_subfolder, os.path.join(new_mask_folder_path, parent_folder_name))
    print("{} was moved to {}".format(parent_folder_name, new_mask_folder_path))