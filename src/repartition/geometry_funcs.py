import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import shortest_path


MIN_RATIO = 1e-6

def find_center_skimg(mask: np.ndarray) -> list[dict]:
    """
        Estimate the center of the tumor bed using the centroid method from scikit-image.

    Parameters
    ----------
    mask : np.ndarray
        mask of the tumor bed.
    Returns
    -------
        centers : list[dict]
    """
    label_map = label(mask)
    centers = []
    for instance in regionprops(label_map):
        instance_mask = np.uint8(label_map == instance.label) * 255
        cy, cx = instance.centroid
        centers.append({'center': (cx, cy), 'mask': instance_mask})
    return centers

def find_center_barycentre(mask: np.ndarray, with_edt=True) -> list[dict]:
    """
        Estimate the center of the tumor bed using the barycenter method with
        a ponderation by the euclidean distance transform.

    Parameters
    ----------
    mask : np.ndarray
        mask of the tumor bed.
    with_edt : bool
        if True, use the euclidean distance transform else, all pixels have a weight of 1.

    Returns
    -------
        centers : list[dict]
    """
    if with_edt:
        mask_u8 = (mask * 255).astype(np.uint8)
        num_labels, label_map = cv2.connectedComponents(mask_u8)
        centers = []

        slices = find_objects(label_map)

        for instance_label in range(1, num_labels): # skip background (0)
            sl = slices[instance_label - 1]
            if sl is None:
                continue

            instance_mask_crop = label_map[sl] == instance_label

            instance_proportion = instance_mask_crop.sum() / label_map.size
            if instance_proportion <= MIN_RATIO:
                continue

            filled_mask = binary_fill_holes(instance_mask_crop)
            if filled_mask is not None:
                filled_mask = filled_mask.astype(np.uint8)

            if with_edt:
                dist = np.zeros_like(filled_mask, dtype=np.float64)
                distance_transform_edt(filled_mask, distances=dist)
            else:
                dist = np.ones_like(filled_mask, dtype=np.float64)

            total  = dist.sum()
            ys, xs = np.nonzero(dist)
            weights = dist[ys, xs]
            cy_crop = int((ys * weights).sum() / total)
            cx_crop = int((xs * weights).sum() / total)

            cy = cy_crop + sl[0].start
            cx = cx_crop + sl[1].start

            full_mask = np.zeros(label_map.shape, dtype=np.uint8)
            full_mask[sl] = instance_mask_crop

            centers.append({
                'center': (cy, cx),
                'mask': full_mask,
            })

        return centers
    else:
        return find_center_moment(mask)

def find_center_enclosing_circle(mask: np.ndarray) -> list[dict]:
    """
        Estimate the center of the tumor bed using the minimum enclosing_circle from opencv.

    Parameters
    ----------
    mask : np.ndarray
        mask of the tumor bed.

    Returns
    -------
        centers : list[dict]
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    num_labels, label_map = cv2.connectedComponents(mask_u8)
    
    centers = []
    for instance_label in range(1, num_labels):  # skip background 0
        instance_mask = (label_map == instance_label).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            instance_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        centers.append({'center': (int(cy), int(cx)), 'mask': instance_mask})
    
    return centers

def find_center_moment(mask: np.ndarray) -> list[dict]:
    """
        Estimate the center of the tumor bed using the moments.

    Parameters
    ----------
    mask : np.ndarray
        mask of the tumor bed.

    Returns
    -------
        centers : list[dict]
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    centers = []
    for idx, cnt in enumerate(contours):
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            instance_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(instance_mask, contours, contourIdx=idx, color=255, thickness=cv2.FILLED)
            centers.append({'center': (cy, cx), 'mask': instance_mask})
    return centers


def find_center_skeleton(mask):
    """
        Estimate the center of the tumor bed by identifying
        the middle point of the skeleton of the tumor bed.

    Parameters
    ----------
    mask : np.ndarray
        mask of the tumor bed.

    Returns
    -------
        centers : list[dict]
    """
    skeleton = skeletonize(mask > 0)
    skel_coords = np.argwhere(skeleton)

    dist_map = np.zeros_like(mask, dtype=np.float64)
    distance_transform_edt(mask, distances=dist_map)
    
    if len(skel_coords) == 0:
        return None

    coord_to_idx = {tuple(coord): i for i, coord in enumerate(skel_coords)}
    num_nodes = len(skel_coords)

    adj = dok_matrix((num_nodes, num_nodes), dtype=np.float32)
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for i, (y, x) in enumerate(skel_coords):
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (ny, nx) in coord_to_idx:
                j = coord_to_idx[(ny, nx)]
                #dist = np.sqrt(dy**2 + dx**2)
                adj[i, j] = 1

    start_idx = 0
    dist_from_start = shortest_path(adj, directed=False, indices=start_idx)
    
    dist_from_start[np.isinf(dist_from_start)] = -1 
    a_idx = np.argmax(dist_from_start)

    dist_from_a = shortest_path(adj, directed=False, indices=a_idx)
    dist_from_a[np.isinf(dist_from_a)] = -1
    max_distance = np.max(dist_from_a)
    # b_idx = np.argmax(dist_from_a)

    half_dist = int(max_distance / 2.0)
    
    valid_mask = dist_from_a >= 0
    valid_indices = np.where(valid_mask)[0]
    
    distances_valid = dist_from_a[valid_indices]
    min_d = np.abs(distances_valid - half_dist).min()
    #idx_closest_to_center = np.argmin(np.abs(distances_valid - half_dist))
    middle_points = valid_indices[np.abs(distances_valid - half_dist) == min_d]

    center_idx = 0
    interior_dist_border = 0
    for p_idx in middle_points:
        row, col = skel_coords[p_idx]
        p_dist_border = dist_map[row, col]
        if p_dist_border > interior_dist_border:
            interior_dist_border = p_dist_border
            center_idx = p_idx

    cy, cx = skel_coords[center_idx]
    mask_arr = (mask * 255).astype(np.uint8)

    return [{
        'center': (int(cy), int(cx)),
        'mask': mask_arr,
    }]
