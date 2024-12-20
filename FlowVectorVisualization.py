import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.omnimotionutil import flow_uv_to_colors

# Allow duplicate libraries (fixes potential environment issues)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path to the folder containing attention maps
folder_path = 'Spatio-temporalAttentionMaps/dog'
attention_maps = []

# Step 1: Load attention maps
for filename in sorted(os.listdir(folder_path)):  # Sort to ensure correct frame order
    if filename.endswith('.png'):
        attention_map = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if attention_map is not None:
            attention_maps.append(attention_map)

# Step 2: Initialize list for flow tensors
flow_tensors = []

# Step 3: Compute flow (delta_x, delta_y) for each pixel
for t in range(len(attention_maps) - 1):
    frame_t, frame_t1 = attention_maps[t], attention_maps[t + 1]
    H, W = frame_t.shape
    flow_tensor = np.zeros((H, W, 2), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            # Extract a neighborhood around the pixel in frame_t1
            neighborhood = frame_t1[max(0, y - 1):min(H, y + 2), max(0, x - 1):min(W, x + 2)]
            max_loc_t1 = np.unravel_index(np.argmax(neighborhood, axis=None), neighborhood.shape)
            delta_x = max_loc_t1[1] - 1  # Horizontal displacement relative to neighborhood center
            delta_y = max_loc_t1[0] - 1  # Vertical displacement relative to neighborhood center
            flow_tensor[y, x, 0] = delta_x
            flow_tensor[y, x, 1] = delta_y
    flow_tensors.append(flow_tensor)

for t in range(len(flow_tensors)):
    frame_t = attention_maps[t]
    flow_tensor = flow_tensors[t]

    H, W = frame_t.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    U, V = flow_tensor[:, :, 0], flow_tensor[:, :, 1]

    plt.figure(figsize=(10, 10))
    plt.imshow(frame_t, cmap='gray')  # Display the original frame as background
    plt.quiver(X, Y, U, V, color='r', scale=1, scale_units='xy', angles='xy')  # Overlay flow vectors
    plt.title(f"Flow Vectors for Frame Pair {t}-{t + 1}")
    plt.axis('off')
    plt.savefig(os.path.join("Flows_dog", f"{t}{t+1}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
