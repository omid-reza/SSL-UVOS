import cv2
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

# Define the file paths for the candidate and target images
# TODO: add 1st pix from all of the folders
candidates_path = ['0.png', '1.png']
# target_path = "DAVIS/Annotations_unsupervised/480p/dog/00000.png"
# for candidate_path in candidates_path:
    # print(get_common_pixels(candidate_path=candidate_path, target_path=target_path))
