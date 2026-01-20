# -*- coding: utf-8 -*-
"""
Created on Jan 8th 2026
@author: Karishma
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_mask(masks, img, path_to_imageseg, img_size=None, save=False, random_color=True):
    """
    Visualize SAM masks over the *actual analyzed image*.

    Default: no resizing (keeps coordinates trustworthy).
    If img_size is provided (w,h), both image and masks are resized consistently.
    """

    # Handle empty detections cleanly
    if masks is None or len(masks) == 0:
        plt.figure(figsize=(8, 8), dpi=150)
        plt.imshow(img)
        plt.axis("off")
        if save:
            plt.savefig(path_to_imageseg, bbox_inches="tight", pad_inches=0)
        plt.close()
        return

    plt.figure(figsize=(8, 8), dpi=150)

    if img_size is not None:
        img_vis = cv2.resize(img, img_size)
    else:
        img_vis = img

    plt.imshow(img_vis)

    for mask in masks:
        m = (mask > 0).astype(np.uint8)
        if img_size is not None:
            m = cv2.resize(m, img_size)
        show_mask(m, plt.gca(), random_color=random_color)

    plt.axis("off")

    if save:
        plt.savefig(path_to_imageseg, bbox_inches="tight", pad_inches=0)

    plt.close()
