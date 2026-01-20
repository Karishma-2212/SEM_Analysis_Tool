# -*- coding: utf-8 -*-
"""
Created on Fri  
@author: karishma
"""

from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def particle_analysis(masks, boxes, pixel_size, path_to_csv_file, save=False):
    Diameter = []
    Area = []

    if masks is None or len(boxes) == 0 or masks.size == 0:
        return pd.DataFrame(columns=["Particle_ID", "Diameter", "Area"])

    n = min(len(boxes), masks.shape[0])

    for i in range(n):
        mask = masks[i, :, :]
        mask = clear_border(mask)

        if np.sum(mask) > 0:
            props = regionprops_table(mask, properties=["area"])
            raw_area = float(props["area"][0])

            A = raw_area * (pixel_size ** 2)
            D = (np.sqrt(A / 3.14159)) * 2

            Area.append(A)
            Diameter.append(D)

    Diameter = np.array([d for d in Diameter if d > 0])
    Area = np.array([a for a in Area if a > 0])

    if len(Diameter) > 0:
        print(f"Mean Diameter: {np.round(np.mean(Diameter), 2)} units")
        print(f"Mean Area: {np.round(np.mean(Area), 2)} units²")

    # histograms
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].hist(Diameter, bins=20, color="#00d4ff", edgecolor="black")
    axs[0].set_title("Diameter Distribution")
    axs[1].hist(Area, bins=20, color="#ff0055", edgecolor="black")
    axs[1].set_title("Area Distribution")
    plt.tight_layout()
    plt.savefig("distribution.png")
    plt.close()

    df = pd.DataFrame({
        "Particle_ID": range(1, len(Diameter) + 1),
        "Diameter": Diameter,
        "Area": Area
    })

    if save:
        df.to_csv(path_to_csv_file, index=False)

    return df
