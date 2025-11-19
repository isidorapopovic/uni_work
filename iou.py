import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.transform import resize


mask_folder = 'masks'
predicted_mask_folder = 'predicted_masks'


mask_files = os.listdir(mask_folder)
predicted_mask_files = os.listdir(predicted_mask_folder)


mask_files.sort()
predicted_mask_files.sort()


mask_files = mask_files[900:1000]
predicted_mask_files = predicted_mask_files[:100]


iou_list = []


for i in range(100):
    
    mask_path = os.path.join(mask_folder, mask_files[i])
    mask = np.array(Image.open(mask_path).resize((256, 256)))

    
    predicted_mask_path = os.path.join(predicted_mask_folder, predicted_mask_files[i])
    predicted_mask = np.array(Image.open(predicted_mask_path).resize((256, 256)))

   
    mask = np.where(mask[:, :, 0] > 0, 1, 0)  
    predicted_mask = np.where(predicted_mask[:, :, 0] > 0, 1, 0)  

    mask = mask > threshold_otsu(mask)
    predicted_mask = predicted_mask > threshold_otsu(predicted_mask)


    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)
    predicted_mask = binary_erosion(predicted_mask, iterations=15)
    predicted_mask = binary_dilation(predicted_mask, iterations=15)


    mask = resize(mask, (256, 256), order=0, preserve_range=True)
    predicted_mask = resize(predicted_mask, (256, 256), order=0, preserve_range=True)


    intersection = np.sum(np.logical_and(mask, predicted_mask))


    union = np.sum(np.logical_or(mask, predicted_mask))


    iou = intersection / union

    iou_list.append(iou)

mean_iou = np.mean(iou_list)
max_iou = np.max(iou_list)

print("Mean IOU:", mean_iou)
print(max_iou)