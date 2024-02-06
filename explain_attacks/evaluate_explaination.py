import numpy as np
from skimage.metrics import structural_similarity
import cv2

def iou_score(image_original, image_attacked):
    """Inserction over Union (IoU) scores

    Args:
        image_original (np.ndarray): clean images used as ground truth
        image_attacked (np.ndarray): attacked images by adversarial attack

    Returns:
        iou (float): iou scores
    """    
    binary_original = (image_original > 0).astype(np.uint8)
    binary_attacked = (image_attacked > 0).astype(np.uint8)

    intersection = np.logical_and(binary_original, binary_attacked)
    union = np.logical_or(binary_original, binary_attacked)

    iou = np.sum(intersection) / np.sum(union)
    
    return iou

def ssim_score(image_original, image_attacked, win_size=7):
    """calculate Structural Similary Index scores

    Args:
        image_original (np.ndarray): clean images used as ground truth
        image_attacked (np.ndarray): attacked images by adversarial attack
        win_size (int, optional): _description_. Defaults to 7.

    Returns:
        ssim_value (float): ssim scores
    """    
    channel_axis = None
    if len(image_original.shape) == 3:
        channel_axis = 2

    ssim_value, _ = structural_similarity(image_original, 
                                          image_attacked, 
                                          full=True, 
                                          win_size=win_size, 
                                          channel_axis=channel_axis, 
                                          data_range=image_original.max() - image_original.min())

    return ssim_value