# Utility functions
#
#
# Author: Marco Huber (marco.huber@igd.fraunhofer.de), Marco Kiesewalter 
# Fraunhofer Institute for Computer Graphics Research IGD, 2025
# 
# https://github.com/marcohuber/recowgnition
# 
# This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
# Copyright (c) 2026 Fraunhofer Institute for Computer Graphics Research IGD
#
import os
import cv2
import json
import numpy as np

def image_iter(path):
    """
    Recursively collect image file paths from a directory.

    Traverses the given directory and all subdirectories, returning a list
    of file paths for all files found.

    Parameters
    ----------
    path : str
        Path to the root directory containing images.
    
    Returns
    -------
    list of str
        A list of full file paths to images found in the directory tree.
    """
    
    image_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:
            if name != "Thumbs.db":
                image_paths.append(os.path.join(path, name))
    
    return image_paths

def load_images(paths):
    """
    Load images from disk and convert them to RGB format.

    Reads each image from the provided file paths using OpenCV and converts
    them from BGR (OpenCV default) to RGB color space.

    Parameters
    ----------
    paths : list of str
        List of file paths to image files.

    Returns
    -------
    list of numpy.ndarray
        List of images as NumPy arrays in RGB format.
    """
    
    images = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images
