import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.omnimotionutil import flow_uv_to_colors, flow_to_image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.set_printoptions(threshold=np.inf)


def save_masked_maps(masked_maps, output_folder):
    """Save masked attention maps to a folder."""
    os.makedirs(output_folder, exist_ok=True)
    for i, masked_map in enumerate(masked_maps):
        output_path = os.path.join(output_folder, f"masked_map_{i+1}.png")
        # Normalize and save the image
        normalized_map = cv2.normalize(masked_map, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(output_path, normalized_map.astype(np.uint8))

folder_path = 'Spatio-temporalAttentionMaps/dog'
attention_maps = []
for filename in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    attention_map = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
    attention_maps.append(attention_map)
attention_maps = np.array(attention_maps)
folder_path = 'Spatio-temporalAttentionMaps/dog_mask'
masks = []
for filename in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    mask = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
    masks.append(mask)

masks = np.array(masks)
masked_attention_maps = []
for attention_map, mask in zip(attention_maps, masks):
    attention_map = attention_map.astype(np.float32)
    mask = mask.astype(np.float32)
    masked_map = attention_map * mask
    masked_attention_maps.append(masked_map)

save_masked_maps(masked_attention_maps, 'Spatio-temporalAttentionMaps/dog_masked')