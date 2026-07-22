# Evaluation Script for the ReCowGnition cow face recognition benchmark
#
#
# Author: Marco Huber (marco.huber@igd.fraunhofer.de), Marco Kiesewalter 
# Fraunhofer Institute for Computer Graphics Research IGD, 2025
# 
# https://github.com/marcohuber/recowgnition
# 
# This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
# Copyright (c) 2026 Fraunhofer Institute for Computer Graphics Research IGD

from utils import image_iter

from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

import os
import cv2
import torch
import numpy as np
import itertools

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### Verification ###


def verification(model, path, not_same_session): 
    """
    Evaluate a biometric verification model by computing similarity scores
    for genuine and impostor image pairs and reporting standard verification
    metrics.

    The function loads images from the given path, generates genuine and
    impostor pairs (optionally enforcing cross-session pairing), computes
    similarity scores using the provided model, saves the scores to disk,
    and reports EER and FNMR at fixed FMR operating points.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to compute similarity scores between image pairs.
    path : str
        Path to the dataset containing images and metadata required to form
        genuine and impostor pairs.
    not_same_session : bool
        If True, only cross-session image pairs are considered.
        If False, image pairs may include samples from the same session.

    Returns
    -------
    eer : float
        Equal Error Rate (EER) of the verification system.
    eer_threshold : float
        Decision threshold corresponding to the EER.
    fnmr10 : float
        False Non-Match Rate (FNMR) at a target False Match Rate (FMR) of 10%.
    fmr10_threshold : float
        Decision threshold corresponding to FMR = 10%.
    fnmr100 : float
        False Non-Match Rate (FNMR) at a target False Match Rate (FMR) of 1%.
    fmr100_threshold : float
        Decision threshold corresponding to FMR = 1%.
    """
    
    
    if not_same_session:
        print("Evaluating: Verification (cross session)...")
    else:
        print("Evaluating: Verification (including same session)...")
        
    images, paths = load_images_trans(path)
    images = torch.stack(images)
    
    gen_pairs, imp_pairs = get_pairs(path, not_same_session)
    gen_scores = compare_image_pairs(model, gen_pairs, images)
    imp_scores = compare_image_pairs(model, imp_pairs, images)
    
    # EER
    eer, eer_threshold = compute_eer(gen_scores, imp_scores)
    
    # FMR
    fnmr10, fmr10_threshold = fnmr_at_fmr(gen_scores, imp_scores, target_fmr=0.1)
    fnmr100, fmr100_threshold = fnmr_at_fmr(gen_scores, imp_scores, target_fmr=0.01)
    
    print(f"EER: {eer:4f}  (threshold = {eer_threshold:.4f})")
    print(f"FNMR @ FMR=10%:  {fnmr10:.4f}  (threshold = {fmr10_threshold:.4f})")
    print(f"FNMR @ FMR=1%:   {fnmr100:.4f}  (threshold = {fmr100_threshold:.4f})")
    print(f"No. of genuine pairs: {len(gen_pairs)}")
    print(f"No. of impostor pairs: {len(imp_pairs)}")
    
    return eer, eer_threshold, fnmr10, fmr10_threshold, fnmr100, fmr100_threshold


###   Identification   ###


def identification_image(model, path, device, top_k=(1,)):
    """
    Evaluate the identification system using cosine similarity,
    reporting Top-k accuracy and the Cumulative Match Characteristic (CMC) curve.

    The function extracts normalized feature embeddings for all images in the
    dataset, performs an all-vs-all identification protocol (including same
    session samples), and ranks gallery identities based on cosine similarity.
    Self-matches are excluded from evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to extract feature embeddings from images.
    path : str
        Path to the dataset containing images and identity information.
    device : torch.device
        Device on which inference is performed (e.g., CPU or CUDA).
    top_k : tuple of int, optional
        Values of k for which Top-k identification accuracy is computed.
        Default is (1,).

    Returns
    -------
    accuracies : dict
        Dictionary mapping each k in `top_k` to the corresponding Top-k
        identification accuracy.
    cmc : numpy.ndarray
        Cumulative Match Characteristic (CMC) curve of length equal to the
        number of unique identities. The value at rank r indicates the
        probability that the correct identity appears within the top (r + 1)
        ranks.
    """
    
    
    print("Evaluating: Identification (including same session)...")
    
    # prepare data
    images, paths = load_images_trans(path)
    _, labels = get_info(paths)
    images = torch.stack(images)

    batch_size = 256
    all_features = []
    
    # get features
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i + batch_size].to(device)
            embs = model(batch)
            embs = F.normalize(embs, p=2, dim=1)
            all_features.append(embs.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = np.array(labels)
    N = len(features)

    # evaluation structures
    top_k = sorted(top_k)
    correct_topk = {k: [] for k in top_k}
    
    unique_gallery_ids = np.unique(labels)
    num_identities = len(unique_gallery_ids)
    cmc = np.zeros(num_identities) 
    
    # calculate similarity
    for i in tqdm(range(N), desc="Computing similarities"):
        sims = cosine_similarity(features[i].reshape(1, -1), features)[0]

        # remove self-comparison
        sims[i] = -np.inf
        
        # sort
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = labels[sorted_idx]

        # identity-based ranking
        _, unique_idx = np.unique(sorted_labels, return_index=True)
        unique_idx = np.sort(unique_idx)
        unique_labels = sorted_labels[unique_idx]

        # top-k accuracy
        for k in top_k:
            correct_topk[k].append(labels[i] in unique_labels[:k])

        # CMC curve
        match_positions = np.where(unique_labels == labels[i])[0]
        if len(match_positions) > 0:
            first_match_rank = match_positions[0]
            cmc[first_match_rank:] += 1


    cmc /= N
    accuracies = {k: np.mean(correct_topk[k]) for k in top_k}

    for k, acc in accuracies.items():
        print(f"Top-{k} Accuracy (same session): "f"{acc:.4f}, No. of Probes: {len(correct_topk[k])}")
        
    return accuracies, cmc



def identification_image_cross_session(model, path, device, top_k=(1,)):
    
    print("Evaluating: Identification (cross session)...")
    
    # prepare data
    images, paths = load_images_trans(path)
    sessions, labels = get_info(paths)
    images = torch.stack(images)
    
    batch_size = 256
    all_features = []
    
    # get features
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            embs = model(batch)          
            embs = F.normalize(embs, p=2, dim=1)
            all_features.append(embs.cpu()) 
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = np.array(labels)
    sessions = np.array(sessions)
    N = len(features)

    # evaluation structures
    unique_gallery_ids = np.unique(labels)
    num_identities = len(unique_gallery_ids)
    cmc = np.zeros(num_identities)
    top_k = sorted(top_k)
    correct_topk = {k: [] for k in top_k}
    valid_probes = 0
    
    for i in tqdm(range(N)):
        
        # skip cows without cross-session genuine
        has_cross_session = np.any((labels == labels[i]) & (sessions != sessions[i]))
        if not has_cross_session:
            continue
        valid_probes +=1

        # compute cosine similarity
        sims = cosine_similarity(
            features[i].reshape(1, -1),
            features
        )[0]

        # Mask invalid comparisons:
        # Only skip where both label AND record are the same & it self
        mask = (labels == labels[i]) & (sessions == sessions[i])
        mask[i] = True
        sims[mask] = -np.inf
        
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = labels[sorted_idx]
        sorted_sessions = sessions[sorted_idx]

        # Keep only first occurrence per identity
        _, unique_idx = np.unique(sorted_labels, return_index=True)
        unique_idx = np.sort(unique_idx)
        unique_labels = sorted_labels[unique_idx]
        unique_sessions = sorted_sessions[unique_idx]
        
        
        # top-k accuracies
        for k in top_k:
           correct_topk[k].append(
               any((unique_labels[j] == labels[i]) and (unique_sessions[j] != sessions[i]) for j in range(min(k, len(unique_labels))))
           )

        # CMC
        match_positions = np.where((unique_labels == labels[i]) & (unique_sessions != sessions[i]))[0]
        if len(match_positions) > 0:
            first_match_rank = match_positions[0]
            cmc[first_match_rank:] += 1

    cmc /= valid_probes
    accuracies = {k: np.mean(correct_topk[k]) if len(correct_topk[k]) > 0 else 0.0 for k in top_k}

    for k, acc in accuracies.items():
        print(f"Top-{k} Accuracy - Cross Session: {acc:.4f}, Probes: {len(correct_topk[k])}")

    return accuracies, cmc



def identification_embedding_fusion_cross_session(model, path, device, top_k=(1,)):
    """
   Evaluate a cross-session identification system using cosine
   similarity, reporting Top-k accuracy and the Cumulative Match Characteristic
   (CMC) curve.

   For each probe image, only gallery images from a *different session* are
   considered valid matches. Probes without any cross-session genuine samples
   are excluded from evaluation.

   Parameters
   ----------
   model : torch.nn.Module
       Trained model used to extract feature embeddings from images.
   path : str
       Path to the dataset containing images, identity labels, and session
       information.
   device : torch.device
       Device on which inference is performed (e.g., CPU or CUDA).
   top_k : tuple of int, optional
       Values of k for which Top-k identification accuracy is computed.
       Default is (1,).

   Returns
   -------
   accuracies : dict
       Dictionary mapping each k in `top_k` to the corresponding Top-k
       cross-session identification accuracy.
   cmc : numpy.ndarray
       Cumulative Match Characteristic (CMC) curve computed over valid probes.
       The value at rank r represents the probability that the correct identity
       appears within the top (r + 1) ranks under the cross-session constraint.
    """
    
    print("Evaluating: Identification (embedding fusion)...")
    
    # prepare data
    images, paths = load_images_trans(path)
    sessions, labels = get_info(paths)
    images = torch.stack(images)

    batch_size = 256
    all_features = []

    # get features
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            embs = model(batch)
            embs = F.normalize(embs, p=2, dim=1)
            all_features.append(embs.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = np.array(labels)
    sessions = np.array(sessions)

    # fusion
    mean_vectors = {}
    for label in np.unique(labels):
        label_mask = labels == label
        for session in np.unique(sessions[label_mask]):
            mask = label_mask & (sessions == session)
            group = features[mask]
            if len(group) > 0:
                mean = group.mean(axis=0)
                mean /= np.linalg.norm(mean) + 1e-12
                mean_vectors[(label, session)] = mean

    if len(mean_vectors) == 0:
        return {k: 0.0 for k in top_k}

    # fused gallery
    keys = list(mean_vectors.keys())
    fused_features = np.stack([mean_vectors[k] for k in keys])
    fused_labels = np.array([k[0] for k in keys])
    fused_sessions = np.array([k[1] for k in keys])
    N = len(fused_features)

    # evaluation structures
    unique_ids = np.unique(fused_labels)
    num_identities = len(unique_ids)
    cmc = np.zeros(num_identities)  
    top_k = sorted(top_k)
    correct_topk = {k: [] for k in top_k}
    valid_probes = 0
    
    # calculate similarity
    for i in tqdm(range(N)):

        # skip probes without cross-session genuine
        has_cross_session = np.any(
            (fused_labels == fused_labels[i]) &
            (fused_sessions != fused_sessions[i])
        )
        if not has_cross_session:
            continue
        valid_probes += 1
        
        # cosine similarity against all fused features
        sims = cosine_similarity(
            fused_features[i].reshape(1, -1),
            fused_features
        )[0]

        # mask invalid comparisons:
        # same label + same session (including self)
        mask = (fused_labels == fused_labels[i]) & \
               (fused_sessions == fused_sessions[i])
        mask[i] = True
        sims[mask] = -np.inf

        # rank descending
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = fused_labels[sorted_idx]
        sorted_sessions = fused_sessions[sorted_idx]

        _, unique_idx = np.unique(sorted_labels, return_index=True)
        unique_idx = np.sort(unique_idx)
        unique_labels = sorted_labels[unique_idx]
        unique_sessions = sorted_sessions[unique_idx]

        for k in top_k:
            correct_topk[k].append(
                any(
                    (unique_labels[j] == fused_labels[i]) and
                    (unique_sessions[j] != fused_sessions[i])
                    for j in range(min(k, len(unique_labels)))
                )
            )
        match_positions = np.where(
            (unique_labels == fused_labels[i]) &
            (unique_sessions != fused_sessions[i])
        )[0]
        
        if len(match_positions) > 0:
            first_match_rank = match_positions[0]
            cmc[first_match_rank:] += 1
        
        
    cmc /= valid_probes
    accuracies = {k: np.mean(correct_topk[k]) if len(correct_topk[k]) > 0 else 0.0 for k in top_k}


    for k, acc in accuracies.items():
        print(f"Top-{k} Accuracy - Video Fusion Cross Session: {acc:.4f}, Probes: {len(correct_topk[k])}")
        
    return accuracies, cmc



def identification_score_fusion_cross_session(model, path, device, top_k=(1,)):
    """
   Evaluate a cross-session identification system using score-level fusion
   of multiple images per identity and session.

   Images are first grouped by (identity, session). For each group, all
   pairwise cosine similarities with other groups are computed and averaged
   to obtain a single fused similarity score. Identification performance is
   then evaluated under a cross-session constraint using Top-k accuracy and
   the Cumulative Match Characteristic (CMC) curve.

   Parameters
   ----------
   model : torch.nn.Module
       Trained model used to extract feature embeddings from images.
   path : str
       Path to the dataset containing images, identity labels, and session
       information.
   device : torch.device
       Device on which inference is performed (e.g., CPU or CUDA).
   top_k : tuple of int, optional
       Values of k for which Top-k identification accuracy is computed.
       Default is (1,).

   Returns
   -------
   accuracies : dict
       Dictionary mapping each k in `top_k` to the corresponding Top-k
       cross-session identification accuracy using score-level fusion.
   cmc : numpy.ndarray
       Cumulative Match Characteristic (CMC) curve computed over valid probes.
       The value at rank r represents the probability that the correct identity
       appears within the top (r + 1) ranks.
    """
    
    print("Evaluating: Identification (score fusion)...")    

    # prepare data
    images, paths = load_images_trans(path)
    sessions, labels = get_info(paths)
    images = torch.stack(images)

    batch_size = 256
    all_features = []

    # get features
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            embs = model(batch)
            embs = F.normalize(embs, p=2, dim=1)
            all_features.append(embs.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = np.array(labels)
    sessions = np.array(sessions)
    
    # fuse
    keys = np.column_stack((labels, sessions))
    unique_keys = np.unique(keys, axis=0)
    group_features = {}

    for key in unique_keys:
        label, session = key
        mask = (labels == label) & (sessions == session)
        group_features[(label, session)] = features[mask]

    keys_list = list(group_features.keys())
    M = len(keys_list)

    score_matrix = np.zeros((M, M), dtype=np.float32)

    # score-level fusion
    
    for i in tqdm(range(M)):
        fi = group_features[keys_list[i]]
        for j in range(M):
            if i == j:
                score_matrix[i, j] = -np.inf
                continue
            fj = group_features[keys_list[j]]
            sims = cosine_similarity(fi, fj)  # all pairwise sims
            score_matrix[i, j] = sims.mean()  # score-level fusion

    labels_array = np.array([k[0] for k in keys_list])
    sessions_array = np.array([k[1] for k in keys_list])

    # evaluation structures
    top_k = sorted(top_k)
    correct_topk = {k: [] for k in top_k}

    unique_ids = np.unique(labels_array)
    num_identities = len(unique_ids)
    cmc = np.zeros(num_identities)
    valid_probes = 0
    
    # evaluate
    for i in range(M):

        # skip probes without cross-session genuine
        has_cross_session = np.any(
            (labels_array == labels_array[i]) & (sessions_array != sessions_array[i])
        )
        if not has_cross_session:
            continue
        valid_probes += 1
        
        sims = score_matrix[i].copy()

        # mask same label + same session (including self)
        mask = (labels_array == labels_array[i]) & (sessions_array == sessions_array[i])
        mask[i] = True
        sims[mask] = -np.inf
        
        # sort descending
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = labels_array[sorted_idx]
        sorted_sessions = sessions_array[sorted_idx]

        # identity-ranking
        _, unique_idx = np.unique(sorted_labels, return_index=True)
        unique_idx = np.sort(unique_idx)
        unique_labels = sorted_labels[unique_idx]
        unique_sessions = sorted_sessions[unique_idx]
      
        # top-k accuracies
        for k in top_k:
           correct_topk[k].append(
               any(
                   (unique_labels[j] == labels_array[i]) and
                   (unique_sessions[j] != sessions_array[i])
                   for j in range(min(k, len(unique_labels)))
               )
           )

        # CMC
        match_positions = np.where(
           (unique_labels == labels_array[i]) &
           (unique_sessions != sessions_array[i])
       )[0]
        if len(match_positions) > 0:
           first_match_rank = match_positions[0]
           cmc[first_match_rank:] += 1

      # ----------------------
    cmc /= valid_probes
    accuracies = {k: np.mean(correct_topk[k]) if len(correct_topk[k]) > 0 else 0.0 for k in top_k}


    for k, acc in accuracies.items():
        print(
            f"Top-{k} Accuracy - Score Fusion Cross Session: "
            f"{acc:.4f}, No. of Probes: {len(correct_topk[k])}"
        )
    
    return accuracies, cmc

###   Utility Functions   ###

def fnmr_at_fmr(genuine_scores: np.ndarray, imposter_scores: np.ndarray, target_fmr: float = 0.01):
    """
    Compute the False Non-Match Rate (FNMR) at the highest threshold
    where the False Match Rate (FMR) does not exceed the target FMR.

    Parameters
    ----------
    genuine_scores : np.ndarray
        Scores assigned to genuine (positive) pairs.
    imposter_scores : np.ndarray
        Scores assigned to imposter (negative) pairs.
    target_fmr : float, optional
        Target FMR (default = 0.01 for 1%).

    Returns
    -------
    fnmr : float
        False Non-Match Rate at FMR <= target.
    threshold : float
        Threshold corresponding to that FMR.
    """

    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # FNMR

    # Find thresholds where FMR <= target
    valid = np.where(fpr <= target_fmr)[0]

    if len(valid) == 0:
        raise ValueError("No threshold achieves FMR <= target")

    idx = valid[np.argmax(tpr[valid])]
    fnmr = fnr[idx]
    threshold = thresholds[idx]

    return fnmr, threshold

def load_images_trans(path: str):
    """
    Load and preprocess all images from a specified directory.

      This function iterates through all image file paths in the given directory,
      reads each image using OpenCV, converts it from BGR to RGB format, applies
      a preprocessing transformation, and returns a list of processed images.

      Parameters
      ----------
      path : str
          Path to the directory containing image files.

      Returns
      -------
      images : list
          List of preprocessed images, where each image is the result of applying
          `transform` to the original image read from the directory.
      paths : list of str
        The original file paths matching the index position of each returned image.
           
    """
    
    images = []
    paths = image_iter(path)
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img)
        images.append(img)
    return images, paths


def compute_eer(genuine_scores: np.ndarray, imposter_scores: np.ndarray):
    """
    Compute the Equal Error Rate (EER) and its corresponding threshold.

    The Equal Error Rate is the point on the ROC curve where the False Acceptance Rate (FAR)
    equals the False Rejection Rate (FRR), i.e., when the false positive rate equals the false
    negative rate. It is commonly used to evaluate biometric or verification systems.

    Parameters
    ----------
    genuine_scores : np.ndarray
        Scores assigned to genuine (positive) pairs. Higher values typically indicate higher similarity.
    imposter_scores : np.ndarray
        Scores assigned to imposter (negative) pairs.

    Returns
    -------
    eer : float
        The Equal Error Rate — the rate at which false acceptances equal false rejections.
    eer_threshold : float
        The score threshold corresponding to the EER.
    """
    
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FPR ≈ FNR
    abs_diffs = np.abs(fnr - fpr)
    idx = np.nanargmin(abs_diffs)
   
    eer_threshold = thresholds[idx]
    eer = fpr[idx]
    
    return eer, eer_threshold


def get_pairs(path: str, not_same_session=True):
    """
    Generate genuine and imposter pairs of images based on their labels.

    This function iterates through all image file paths in a given directory,
    extracts identity labels from filenames, and forms all possible unique pairs.
    It then classifies each pair as either 'genuine' (same identity) or 'imposter'
    (different identities).

    Parameters
    ----------
    path : str
        Path to the directory containing image files.
        
    Returns
    -------
    gen_pairs : list[tuple[int, int]]
        List of index pairs corresponding to genuine (same identity) matches.
    imp_pairs : list[tuple[int, int]]
        List of index pairs corresponding to imposter (different identity) matches.
     """
    
    
    labels = []
    paths = image_iter(path)

    for p in paths:
        filename = p.split("/")[-1]
        parts = filename.split("_")
        label = parts[0] + "_" + parts[1]
        labels.append(label)

    comb = list(itertools.combinations(range(len(labels)), 2))
    gen_pairs, imp_pairs = check_pair(labels, comb, not_same_session)
    return gen_pairs, imp_pairs

def check_pair(labels, comb, not_same_session): 
    """
    Separate pairs of indices into genuine and imposter pairs based on labels.

    Given a list of labels corresponding to items (e.g., images) and a list of
    index pairs, this function classifies each pair as either a genuine pair
    (both items have the same label) or an imposter pair (items have different labels).

    Parameters
    ----------
    labels : list[str]
        A list of labels corresponding to the items. Each label must follow the
        format ``sessionID_IDlabel``.
    comb : list[tuple[int, int]]
        A list of index pairs (tuples) referring to positions in the ``labels`` list.
    not_same_session : bool
        If True, genuine pairs where both items belong to the same session are
        discarded. If False, all genuine pairs are included regardless of session.


    Returns
    -------
    genuine_pairs : list[tuple[int, int]]
        List of index pairs where both items have the same label.
    imposter_pairs : list[tuple[int, int]]
        List of index pairs where items have different labels.
    """
    
    genuine_pairs = []
    imposter_pairs = []
    
    for p, r in comb:
        session_p, id_p = labels[p].split("_", 1)
        session_r, id_r = labels[r].split("_", 1)

        if id_p == id_r:
            if not_same_session:
                if session_p != session_r:
                    genuine_pairs.append((p, r))
            else:
                genuine_pairs.append((p, r))
        else:
            imposter_pairs.append((p, r))

    return genuine_pairs, imposter_pairs

def compare_image_pairs(model, pairs, images, image_batch_size=256, pair_batch_size=100_000, device="cuda"):
    """
    Compute cosine similarity scores for a list of image index pairs using a
    trained embedding model.

    The function first extracts L2-normalized embeddings for all images in
    batches, then computes cosine similarity scores for the specified image
    pairs in large batches to reduce overhead.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to extract feature embeddings from images.
    pairs : list of tuple of int
        List of (probe_index, reference_index) pairs for which similarity
        scores are computed.
    images : torch.Tensor
        Tensor containing all images in the dataset, indexed consistently
        with the indices in `pairs`.
    image_batch_size : int, optional
        Batch size used when computing image embeddings.
        Default is 256.
    pair_batch_size : int, optional
        Number of image pairs processed per batch when computing similarity
        scores. Default is 100000.
    device : str or torch.device, optional
        Device on which inference is performed (e.g., "cuda" or "cpu").
        Default is "cuda".

    Returns
    -------
    scores : list of float
        Cosine similarity scores for each image pair, in the same order as
        the input `pairs`.
    """
    
    model = model.to(device).eval()
    embeddings = []

    # Compute embeddings & normalize
    with torch.no_grad():
        for i in tqdm(range(0, len(images), image_batch_size)):
            batch = images[i:i+image_batch_size].float().to(device)
            embs = model(batch)
            embs = F.normalize(embs, p=2, dim=1)
            embeddings.append(embs.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    # Compute similarity
    scores = []
    for i in tqdm(range(0, len(pairs), pair_batch_size)):
        batch_pairs = pairs[i:i+pair_batch_size]
        p_idx = torch.tensor([p for p, r in batch_pairs])
        r_idx = torch.tensor([r for p, r in batch_pairs])
        s = F.cosine_similarity(
            embeddings[p_idx],
            embeddings[r_idx]
        )
        scores.extend(s.tolist())

    return scores


def get_info(paths):
    """
    Extract session IDs and identity labels from file paths.

    Filenames are expected to follow the format ``sessionID_label`` (without
    file extension). The session ID and label are extracted from each filename
    and returned as separate lists aligned with the input paths.

    Parameters
    ----------
    paths : list[str]
    A list of file paths. Only the filename (without directory and extension)
    is used to extract session and label information.

    Returns
    -------
    sessions : list[str]
        List of session identifiers extracted from the filenames.
    labels : list[str]
        List of identity labels extracted from the filenames.
    """
    
    sessions = []
    labels = []

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        parts = name.split("_")
        sess = parts[0]
        label = parts[1]
        sessions.append(sess)
        labels.append(label)

    return sessions, labels
