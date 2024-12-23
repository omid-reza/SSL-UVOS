import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.omnimotionutil import flow_uv_to_colors, flow_to_image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

folder_path = 'Spatio-temporalAttentionMaps/dog'
attention_maps = []
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        attention_map = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if attention_map is not None:
            attention_maps.append(attention_map)
print("attention_maps is loaded.")
flow_tensors = []

# Step 3: Compute the displacement between consecutive frames
for t in range(len(attention_maps) - 1):
    print("Processing frame {}".format(t))
    frame_t, frame_t1 = attention_maps[t], attention_maps[t + 1]
    H, W = frame_t.shape
    max_loc_t1 = np.unravel_index(np.argmax(frame_t1, axis=None), frame_t1.shape)
    max_y, max_x = max_loc_t1
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    delta_x = max_x - x_coords
    delta_y = max_y - y_coords
    flow_tensor = np.stack((delta_x, delta_y), axis=-1).astype(np.float32)
    flow_tensors.append(flow_tensor)

flow_tensors = np.array(flow_tensors)
for t in range(flow_tensors.shape[0]):
    # flow_image  = flow_to_image(flow_tensors[t])
    flow_image = flow_uv_to_colors(flow_tensors[t, :, :, 0], flow_tensors[t, :, :, 1])
    plt.figure(figsize=(8, 8))
    plt.imshow(flow_image)
    plt.title(f"Flow Visualization for Frame Pair {t}-{t + 1}")
    plt.axis('off')
    plt.savefig(os.path.join("Flows_dog", f"{t}{t + 1}.png"), bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

import cv2
import os

image_folder = "Flows_dog"
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(x.split('.')[0]))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(os.path.join("Flows_dog", "Flow"+".avi"), 0, 4, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()