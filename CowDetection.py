# CowDetect: Cow Face Detection 
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
# 


import cv2
import os
import argparse
import numpy as np

from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

from utils import image_iter, load_images

# pre-set Muzzle position
mean_muzzle_pos = (int(56), int(98))
mean_muzzle_size = (int(25), int(15)) 


def extract_frames(video_path: str, frame_step: int = 1, rotation: str = None, crops: list[int] = None):
    """
    Extract frames from a video file with optional rotation and cropping.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    frame_step : int, optional
        Interval between extracted frames.
        For example, `1` extracts every frame, `2` extracts every other frame, etc.
        Default is `1`.
    rotation : str or None, optional
        Rotation mode to apply to each extracted frame using OpenCV's predefined rotate flags.
        Accepted values are:
            - `'90_clockwise'`
            - `'90_counterclockwise'`
            - `'180'`
            - `'45'`
            - `None` (no rotation)
        Default is `None`.
        This can be used to adust the camera angle based on the setup.
    crops : list of int, optional
        List of four integers `[top, bottom, left, right]` specifying 
        the number of pixels to crop from each edge of the frame.
        Default is `[0, 0, 0, 0]`.
        This can be used to avoid unwanted detections in a fixed camera setup.

    Returns
    -------
    frames : list of numpy.ndarray
        List of frames (as NumPy arrays) extracted from the video after
        applying the optional rotation and cropping.
    fids : list of str
        List of unique frame identifiers generated using the frame count.
    """
    
    if crops is None:
        crops = [0, 0, 0, 0]


    rotation_flags = {
        "90_clockwise": cv2.ROTATE_90_CLOCKWISE,
        "90_counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
     
    if rotation == "None":
        rotation = None
        
    if rotation is not None and rotation not in rotation_flags and rotation != "45":
        valid_options = list(rotation_flags.keys()) + ["45", "None"]
        raise ValueError(f"Invalid rotation value '{rotation}'. Valid options are {valid_options}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    frames = []

    fid = get_id(video_path)
    fids = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_step == 0:
            # Apply rotation if specified
            if rotation is not None and rotation in rotation_flags:
              frame = cv2.rotate(frame, rotation_flags[rotation])
            elif rotation is not None and rotation == "45":
              frame = rotate_45(frame)
              
            # Apply cropping
            height, width = frame.shape[:2]
            frame = frame[crops[0]:height - crops[1], crops[2]:width - crops[3]]
            
            frames.append(frame)
            fids.append(f"{fid}_{str(frame_count)}")
            
        frame_count += 1
    
    cap.release()
    return frames, fids


def detect(model_path: str, frames: list[np.ndarray], fids: list[str], threshold_cow: float, threshold_muzzle: float):
    """
    Detect cattle and their muzzles in a sequence of frames using a YOLO model.

    Parameters
    ----------
    model_path : str
        Path to the YOLO model file (e.g., a `.pt` or `.onnx` checkpoint).
    frames : list of numpy.ndarray
        List of video frames (as NumPy arrays) to perform detection on.
    threshold_cow : float
        Minimum confidence threshold for detecting cattle.
        Detections below this confidence level are ignored.
    threshold_muzzle : float
        Minimum confidence threshold for detecting muzzles associated with a cow.
        Detections below this confidence level are ignored.
        
    Returns
    -------
    crop_frames : list of tuple
        A list of tuples, where each tuple contains:
            - `crop` (numpy.ndarray): The cropped image region containing the detected cow.
            - `cid` (str): A unique crop ID.
            - `bbox` (list of int): The bounding box coordinates `[x1, y1, x2, y2]` of the cow.
            - `muzzles` (list of int): The bounding box coordinates `[x1, y1, x2, y2]` of the muzzle.
    
    """
    
    model = YOLO(model_path)
    crop_frames = []

    # Process each frame
    for idx, frame in enumerate(frames):
        result = model(frame, verbose=False, show=False)
        
        # Iterate over all detections
        for r in result:
            for jdx, box in enumerate(r.boxes):

                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Skip non-cattle or low-confidence detections
                if label != "Cattle" or conf < threshold_cow:
                    continue

                # Find associated muzzles
                muzzles = find_muzzles(result, (x1, y1, x2, y2), threshold_muzzle, model.names)
                if not muzzles:
                    continue

                # Crop the detected area
                crop = frame[y1:y2, x1:x2]

                # Filter by aspect ratio to neglect non-useable detections
                h, w = crop.shape[:2]
                
                if h ==0 :
                    continue

                aspect_ratio = w / h
                if aspect_ratio > 1.2:
                    continue

                if fids == None:
                    cid = str(jdx) 
                else:
                    cid = f"{fids[idx]}_{str(jdx)}" 
                   
                crop_frames.append((crop, cid, [x1, y1, x2, y2], muzzles))

    return crop_frames


def find_muzzles(predictions: list, cattle_bounds: tuple[int, int, int, int], threshold_muzzle: float, names: dict[int, str]):
    """
    Find muzzle detections that lie within a given cattle bounding box.

    Parameters
    ----------
    predictions : list of predictions
        YOLO model predictions for one or more frames.
    cattle_bounds : tuple of int
        Bounding box coordinates of the detected cow in the format `(x1, y1, x2, y2)`.
    threshold_muzzle : float
        Minimum confidence threshold for muzzle detections.
    names : dict of int to str
        Mapping of class indices to class names from the YOLO model.

    Returns
    -------
    muzzles : list of list of int
        A list of muzzle bounding boxes that lie completely within the given cow’s bounds.
        Each bounding box is represented as `[x1, y1, x2, y2]`.
    """
    
    x1_cow, y1_cow, x2_cow, y2_cow = cattle_bounds
    muzzles = []

    # Iterate over all detections
    for res in predictions:
        for b in res.boxes:

            # Confidence and class
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            label = names[cls]

            # Skip non-muzzle or low-confidence detections
            if label != "Muzzle" or conf < threshold_muzzle:
                continue

            # Bounding box coordinates
            x1_m, y1_m, x2_m, y2_m = b.xyxy[0].cpu().numpy().astype(int)

            # Keep only muzzles fully inside the cattle bounding box
            if x1_m < x1_cow or x2_m > x2_cow or y1_m < y1_cow or y2_m > y2_cow:
                continue

            muzzles.append([x1_m, y1_m, x2_m, y2_m])

    return muzzles
    
def get_id(path):
    """
    Extract an identifier from a file path.

    The identifier is defined as the file name without its directory
    path or file extension.

    Args:
        path (str): Path to the file.

    Returns:
        str: File name without the extension.
    """
    
    fid = os.path.splitext(os.path.basename(path))[0]
    return fid

def align(crops, mean_pos, mean_size):
    """
    Align cropped cattle images by normalizing muzzle position and scale.

    For each input image, the function rescales the crop so that the muzzle
    width matches a reference mean size, then translates the image so the
    muzzle center is placed at a reference mean position. The result is a
    fixed-size (112x112) aligned image.

    Args:
        crops (list): A list of tuples of the form
            (crop, fid, cattle, muzzles), where:
            - crop (np.ndarray): Cropped RGB image.
            - fid (str): File or image identifier.
            - cattle (list or tuple): Bounding box of the cattle
              [x1, y1, x2, y2].
            - muzzles (list): List of muzzle bounding boxes
              [x1, y1, x2, y2] in original image coordinates.
        mean_pos (tuple): Target (x, y) position for the muzzle center
            in the aligned image.
        mean_size (tuple): Reference muzzle size (width, height); only
            the width is currently used for scaling.

    Returns:
        list: A list of tuples (aligned_image, fid), where:
            - aligned_image (PIL.Image.Image): The aligned 112x112 RGB image.
            - fid (str): File or image identifier.
    """
    
    
    aligned_crops = []
    for img in crops:
        crop, fid, cattle, muzzles = img

        for muzzle in muzzles:
            # Compute muzzle position after cropping
            muzzle[0] -= cattle[0]
            muzzle[2] -= cattle[0]
            muzzle[1] -= cattle[1]
            muzzle[3] -= cattle[1]

            # Compute the center
            x_center = (muzzle[0] + muzzle[2]) / 2
            y_center = (muzzle[1] + muzzle[3]) / 2

            # Get the scale values for resizing
            width_muzzle = muzzle[2] - muzzle[0]
            scale_by = mean_size[0] / width_muzzle
            x_center *= scale_by
            y_center *= scale_by

            # Resize the image while maintaining aspect ratio
            size = (int(crop.shape[1] * scale_by), int(crop.shape[0] * scale_by))
            resized_crop = Image.fromarray(crop)
            resized_crop = resized_crop.resize(size, Image.LANCZOS)
            
            # Compute values where to position image to get muzzle at mean position
            x = int(mean_pos[0] - x_center)
            y = int(mean_pos[1] - y_center)

            new_img = Image.new("RGB", (112, 112), (0, 0, 0))
            new_img.paste(resized_crop, (x, y))
            aligned_crops.append((new_img, fid))
            
    return aligned_crops


def save_crops(crops: list, save_path: str, prefix: str = "crop", verbose: bool = False, mode: str = "video"):
    """
    Save cropped cow faces to disk as JPEG files.

    Parameters
    ----------
    crops : list of crops
        list of cropped cow faces
    save_path : str
        Directory path where the cropped images will be saved. Created if it doesn't exist.
    prefix : str, optional
        Prefix for saved file names. Default is 'crop'.
    verbose : bool, optional
        If True, prints the saved file paths. Default is False.

    Returns
    -------
    None
        Saves each cropped image as a JPEG file in `save_path` with filenames like `prefix_0.jpg`.   
    
    """
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, crop_img in enumerate(crops):
        file_name = f"{crop_img[1]}_{idx}.jpg"
        file_path = os.path.join(save_path, file_name)
        img = crop_img[0]
        if mode == "images":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, np.array(img))
        if verbose:
            print(f"Saved crop: {file_path}")
           

def rotate_45(frame):
    """
    Utility function to add rotation by 45 degree.

    Parameters
    ----------
    frame : (np.ndarray): RGB image
        
    Returns
    -------
    rotated
        np.ndarray: same frame rotated by 45 degree. 
    
    """
    (h, w) = frame.shape[:2]
    center = (w / 2, h / 2)
    
    # +45 degrees = counter-clockwise
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(frame, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    return rotated


def main():
    
    # Parse
    parser = argparse.ArgumentParser(
        description="Cow Detection Pipeline"
    )
    parser.add_argument("--mode", required=True, type=str, choices=["detect_video", "detect_images", "showcase", "create_dataset"], help="Modes: 'detect_video', 'detect_images or 'create_dataset'")
    parser.add_argument("--input", required=True, help="Path to input video file or image folder.")
    parser.add_argument("--output", required=True, help="Path to save annotated video or crops.")
    parser.add_argument("--model", required=True, help="Path to cow detection YOLO model file.")
    parser.add_argument("--thr_c", type=float, default=0.5, help="Cow face confidence threshold.")
    parser.add_argument("--thr_m", type=float, default=0.2, help="Muzzle confidence threshold.")
    parser.add_argument("--frame_step", type=int, default=1, help="Frame extraction interval.")
    parser.add_argument("--rotation",type=str,
        choices=["90_clockwise", "90_counterclockwise", "180", "None", "45"], default="None",
        help="Rotation mode for frames: '90_clockwise', '90_counterclockwise', '180','45' or 'None'.")
    parser.add_argument("--crops", nargs=4, type=int, default=[0, 0, 0, 0], help="Crop margins: top bottom left right.")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose mode to output file paths.")
    parser.add_argument("--prefix", type=str, default="crop", help="Prefix of extracted frames.")
   
    args = parser.parse_args()
    
    # Run detect video
    if args.mode == "detect_video":
        frames, fids = extract_frames(video_path=args.input, frame_step=args.frame_step, rotation=args.rotation, crops=args.crops)
        detected = detect(model_path=args.model, frames=frames, fids=fids, threshold_cow=args.thr_c, threshold_muzzle=args.thr_m)
        save_crops(detected, save_path=args.output, prefix=args.prefix, verbose=args.verbose, mode="video")
    
    # Run detect images
    elif args.mode == "detect_images":
        image_path = image_iter(args.input)
        images = load_images(image_path)
        detected = detect(model_path=args.model, frames=images, fids=None, threshold_cow=args.thr_c, threshold_muzzle=args.thr_m)
        save_crops(detected, save_path=args.output, prefix=args.prefix, verbose=args.verbose, mode="images")
        
    # Run create dastaset that includes alignment
    elif args.mode == "create_dataset":
        video_paths = image_iter(args.input)
        for video_path in tqdm(video_paths):
            frames, fids = extract_frames(video_path=video_path, frame_step=args.frame_step, rotation=args.rotation, crops=args.crops)
            detected = detect(model_path=args.model, frames=frames, fids=fids, threshold_cow=args.thr_c, threshold_muzzle=args.thr_m)
            mean_pos, mean_size = mean_muzzle_pos, mean_muzzle_size
            aligned_detected = align(detected, mean_pos, mean_size)
            save_crops(aligned_detected, save_path=args.output, prefix=args.prefix, verbose=args.verbose, mode="video")
    
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Valid options are 'detect_video', 'detect_images' or 'create_dataset'.")
    
    
if __name__ == "__main__":
   main()
   
