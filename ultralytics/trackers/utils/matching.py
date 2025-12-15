# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (np.ndarray): Array of matched indices of shape (K, 2), where K is the number of matches.
        unmatched_a (np.ndarray): Array of unmatched indices from the first set, with shape (L,).
        unmatched_b (np.ndarray): Array of unmatched indices from the second set, with shape (M,).

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Height Modulated IoU (HMIoU) between tracks.

    HMIoU = IoU * HIoU, where HIoU is IoU along the vertical axis (height),
    as in Hybrid-SORT (Yang et al.).
    """
    print(">>> Using HMIoU iou_distance")

    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # For Ultralytics tracks, xyxy gives [x1, y1, x2, y2]
        atlbrs = [track.xywha if getattr(track, "angle", None) is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if getattr(track, "angle", None) is not None else track.xyxy for track in btracks]

    if not atlbrs or not btlbrs:
        return np.empty((len(atlbrs), len(btlbrs)), dtype=np.float32)

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)

    # Rotated boxes (len == 5): fall back to original probabilistic IoU
    if atlbrs.shape[1] == 5 and btlbrs.shape[1] == 5:
        ious = batch_probiou(atlbrs, btlbrs).numpy()
        return 1.0 - ious

    # ----------- Axis-aligned case: compute HMIoU -----------
    # 1) Standard IoU (area-based)
    ious = bbox_ioa(atlbrs, btlbrs, iou=True)  # shape [N_tracks, N_dets]

    # 2) Height IoU (HIoU) along vertical axis
    N, M = ious.shape
    h_ious = np.zeros_like(ious, dtype=np.float32)

    # Extract y1,y2 for all boxes (assumes [x1, y1, x2, y2])
    y1_a = atlbrs[:, 1].reshape(N, 1)  # [N,1]
    y2_a = atlbrs[:, 3].reshape(N, 1)
    y1_b = btlbrs[:, 1].reshape(1, M)  # [1,M]
    y2_b = btlbrs[:, 3].reshape(1, M)

    inter_h = np.minimum(y2_a, y2_b) - np.maximum(y1_a, y1_b)
    inter_h = np.clip(inter_h, a_min=0.0, a_max=None)  # no negative overlap

    union_h = np.maximum(y2_a, y2_b) - np.minimum(y1_a, y1_b)
    # avoid division by zero
    valid = union_h > 0
    h_ious[valid] = (inter_h[valid] / union_h[valid]).astype(np.float32)

    # 3) Height Modulated IoU
    hmious = ious * h_ious  # element-wise

    # 4) Convert to cost: distance = 1 - HMIoU
    return 1.0 - hmious


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks, where each track contains embedding features.
        detections (list[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
