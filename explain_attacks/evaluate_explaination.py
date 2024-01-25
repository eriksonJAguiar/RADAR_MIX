import numpy as np
from skimage.metrics import structural_similarity
import cv2

def iou_score(image_original, image_attacked):
    
    binary_original = (image_original > 0).astype(np.uint8)
    binary_attacked = (image_attacked > 0).astype(np.uint8)

    intersection = np.logical_and(binary_original, binary_attacked)
    union = np.logical_or(binary_original, binary_attacked)

    iou = np.sum(intersection) / np.sum(union)
    
    return iou

def ssim_score(image_original, image_attacked):
    
    print(image_original.shape)
    if len(image_original.shape) == 3:
        image_original = cv2.cvtColor(image_original.transpose(3, 1, 2), cv2.COLOR_BGR2GRAY)
    if len(image_attacked.shape) == 3:
        image_attacked = cv2.cvtColor(image_attacked.transpose(3, 1, 2), cv2.COLOR_BGR2GRAY)
    
    ssim_value, _ = structural_similarity(image_original, image_attacked, full=True)

    return ssim_value