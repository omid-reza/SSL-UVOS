import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.omnimotionutil import flow_uv_to_colors, flow_to_image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.set_printoptions(threshold=np.inf)

folder_path = 'Spatio-temporalAttentionMaps/dog_masked'
attention_maps = []
for filename in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
    attention_map = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
    attention_maps.append(attention_map)
flow_tensors = []

# Step 3: Compute the displacement between consecutive frames
for t in range(len(attention_maps) - 1):
    frame_t, frame_t1 = attention_maps[t], attention_maps[t + 1]
    H, W = frame_t.shape
    flow = cv2.calcOpticalFlowFarneback(
        prev=frame_t,
        next=frame_t1,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_tensors.append(flow)  # Shape(H, W, 2) where each pixel has (dx, dy)

for i in range(len(flow_tensors)):
    flow = flow_tensors[i]
    flow = flow_uv_to_colors(flow[:, :, 0], flow[:, :, 1])
    # plt.figure(figsize=(8, 8))
    # plt.imshow(flow)
    # plt.title(f"Flow Visualization for Frame Pair {i}-{i + 1}")
    # plt.axis('off')
    plt.imsave(os.path.join("Flows_dog", f"{i}{i + 1}.png"), flow)



# flow_tensors = np.array(flow_tensors)
# flow_images = flow_to_image(flow_tensors)
# for t in range(flow_tensors.shape[0]):
#     flow_image = flow_images[t]
#     plt.figure(figsize=(8, 8))
#     plt.imshow(flow_image)
#     plt.title(f"Flow Visualization for Frame Pair {t}-{t + 1}")
#     plt.axis('off')
#     plt.savefig(os.path.join("Flows_dog", f"{t}{t + 1}.png"), bbox_inches='tight', pad_inches=0)
#     plt.cla()
#     plt.close()

# import cv2
# import os
#
# image_folder = "Flows_dog"
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# images.sort(key=lambda x: int(x.split('.')[0]))
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
# video = cv2.VideoWriter(os.path.join("Flows_dog", "Flow"+".avi"), 0, 4, (width,height))
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
# cv2.destroyAllWindows()
# video.release()