import cv2
import os

categories = [x[0] for x in os.walk("Spatio-temporalAttentionMaps")]

for category in categories:
    image_folder = os.path.join("Spatio-temporalAttentionMaps", category)
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('.')[0]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join("Spatio-temporalAttentionMapsVideos", category+".avi"), 0, 4, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()