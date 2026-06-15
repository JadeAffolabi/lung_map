import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects
from scipy.spatial import distance
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

######################################################
# Il reste la normalisation de la distance tumor-bed_center !

######################################################
MIN_RATIO = 1e-6

def find_center_skimg(mask):
    label_map = label(mask)
    centers = []
    for instance in regionprops(label_map):
        instance_mask = np.uint8(label_map == instance.label) * 255
        cy, cx = instance.centroid
        centers.append({'center': (cx, cy), 'mask': instance_mask})
    return centers

def find_center_edt_barycentre(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    num_labels, label_map = cv2.connectedComponents(mask_u8)
    centers = []

    # bounding boxes de chaque label
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
        dist = np.zeros_like(filled_mask, dtype=np.float64)
        distance_transform_edt(filled_mask, distances=dist)

        # Barycentre
        total  = dist.sum()
        ys, xs = np.nonzero(dist)
        weights = dist[ys, xs]
        cy_crop = int((ys * weights).sum() / total)
        cx_crop = int((xs * weights).sum() / total)

        # Remettre dans les coordonnées du masque original
        cy = cy_crop + sl[0].start
        cx = cx_crop + sl[1].start

        full_mask = np.zeros(label_map.shape, dtype=np.uint8)
        full_mask[sl] = instance_mask_crop

        centers.append({
            'center': (cy, cx),
            'mask': full_mask,
        })

    return centers

def get_components(mask):
    mask_u8 = (mask * 255).astype(np.uint8)
    num_labels, label_map = cv2.connectedComponents(mask_u8)
    components = []

    # bounding boxes de chaque label
    slices = find_objects(label_map)

    for instance_label in range(1, num_labels): # skip background (0)
        sl = slices[instance_label - 1]
        if sl is None:
            continue

        instance_mask_crop = label_map[sl] == instance_label

        instance_proportion = instance_mask_crop.sum() / label_map.size
        if instance_proportion <= MIN_RATIO:
            continue

        full_mask = np.zeros(label_map.shape, dtype=np.uint8)
        full_mask[sl] = instance_mask_crop

        components.append({
            'mask': full_mask,
        })

    return components

def find_center_enclosing_circle(mask):
    num_labels, label_map = cv2.connectedComponents(mask)
    
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
        centers.append({'center': (int(cx), int(cy)), 'mask': instance_mask})
    
    return centers

def find_center_moment(mask):
    contours, _ = cv2.findContours(
        mask,
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
            centers.append({'center': (cx, cy), 'mask': instance_mask})
    return centers

def get_objects_center(seg_masks, obj_label, find_center):

    mask = seg_masks[obj_label]
    mask = (mask * 255).astype(np.uint8)

    centers = find_center(mask)

    return centers

def get_tumor_bed_center(seg_masks):
    tumor_bed_mask = (seg_masks['tumor_bed'] > 0).astype(np.uint8)
    center = find_center_edt_barycentre(tumor_bed_mask)[0]['center']
    return center

def compute_distance_to_bed_center(bed_center, tumors_infos):
    for tumor in tumors_infos:
        tumor_center = tumor['center']
        dist_to_bed_center = distance.euclidean(bed_center, tumor_center)
        tumor.update({'distance-center': dist_to_bed_center})
    return tumors_infos

def compute_distance_to_border(seg_masks, tumors_infos, ref='tumor_bed'):
    mask = (seg_masks[ref]*255).astype(np.uint8)

    dist_map = distance_transform_edt(mask)

    for tumor in tumors_infos:
        if dist_map is not None:
            dist_to_border = dist_map[tumor['mask'] > 0].min()
            tumor.update({'distance-border': dist_to_border})
        else:
            return None

    return tumors_infos

def compute_area_percentage(seg_masks, tumor_infos):
    total_tumor_pixels = np.count_nonzero(seg_masks['tumor'])
    total_bed_pixels = np.count_nonzero(seg_masks['tumor_bed'])
    for tumor in tumor_infos:
        tumor_pixels = np.count_nonzero(tumor['mask'])
        tumor.update({'area': tumor_pixels / total_bed_pixels})

    return tumor_infos, total_tumor_pixels / total_bed_pixels

def get_histogram(seg_masks, label, ref="border", bed_center=None, nb_bin=64):
    if ref == 'border':
        dist_map = distance_transform_edt(seg_masks['tumor_bed'])
        if dist_map is not None:
            dist_map_filtered = dist_map[seg_masks[label]]
            hist, bins = np.histogram(dist_map_filtered, bins=nb_bin, density=True)
            return hist, bins
        else:
            return None
    elif ref == 'center':
        assert bed_center is not None, "bed_center should not be None."
        tumors_pix_coord = np.argwhere(seg_masks[label])
        dist_to_bed_center = [distance.euclidean(bed_center, coord) for coord in tumors_pix_coord]
        hist, bins = np.histogram(dist_to_bed_center, bins=nb_bin, density=True)
        return hist, bins
    else:
        raise NotImplementedError(
                f"""unknown reference '{ref}', expected one of:
                 border, center."""
            )
