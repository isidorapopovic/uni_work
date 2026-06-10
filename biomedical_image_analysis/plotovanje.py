import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


mask_folder = 'masks'
predicted_mask_folder = 'predicted_masks'


mask_files = os.listdir(mask_folder)
predicted_mask_files = os.listdir(predicted_mask_folder)


mask_files.sort()
predicted_mask_files.sort()


mask_files = mask_files[900:1000]
predicted_mask_files = predicted_mask_files[:100]


fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Loop through the mask files
for i in range(4):
    
    mask_path = os.path.join(mask_folder, mask_files[i])
    mask = np.array(Image.open(mask_path).resize((256, 256)))

    
    predicted_mask_path = os.path.join(predicted_mask_folder, predicted_mask_files[i])
    predicted_mask = np.array(Image.open(predicted_mask_path).resize((256, 256)))

    
    axs[0, i].imshow(mask, cmap='gray')
    axs[0, i].set_title('True Mask')

    
    axs[1, i].imshow(predicted_mask, cmap='gray')
    axs[1, i].set_title('Predicted Mask')


fig.tight_layout()

plt.show()