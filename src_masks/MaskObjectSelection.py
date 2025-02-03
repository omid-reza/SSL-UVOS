import os
import cv2
import numpy as np


def compute_iou(mask1, mask2):
    mask1 = cv2.resize(mask1, (mask2.shape[1], mask2.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def find_best_matching_folder(reference_mask_path, folders_root):
    reference_mask = cv2.imread(reference_mask_path, cv2.IMREAD_GRAYSCALE)
    reference_mask = reference_mask > 0  # Convert to binary
    best_iou, best_folder = 0, None

    for folder in os.listdir(folders_root):
        folder_path = os.path.join(folders_root, folder)
        first_frame_mask_path = os.path.join(folder_path, "00000.png")
        if os.path.exists(first_frame_mask_path):
            candidate_mask = cv2.imread(first_frame_mask_path, cv2.IMREAD_GRAYSCALE)
            candidate_mask = candidate_mask > 0
            iou = compute_iou(reference_mask, candidate_mask)
            if iou > best_iou:
                best_iou = iou
                best_folder = folder
    return best_folder, best_iou

reference_mask_path = "DAVIS/Annotations_unsupervised/480p/dog/00000.png"
folders_root = "masks/dog_split/"
best_folder, best_iou = find_best_matching_folder(reference_mask_path, folders_root)
print(f"Best matching folder: {best_folder} with IoU: {best_iou:.4f}")
