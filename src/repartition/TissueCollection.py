
import cv2
import numpy as np
from scipy.ndimage import find_objects
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance
from scipy import stats
import textwrap
import matplotlib.pyplot as plt
import math


DOWNSAMPLE_FACTOR = 128
ORIGINAL_MICRO_PER_PIXEL = 0.2425
PIXEL_SIZE = ORIGINAL_MICRO_PER_PIXEL * DOWNSAMPLE_FACTOR
MIN_RATIO = 1e-6

class TissueCollection:

    def __init__(self, name, mask) -> None:
        self.name = name
        self.mask = mask
        self._tissues = self._get_components(mask)
        self.is_tumoral = 'tumor' in name.lower()
        self._border_dist_map = None

    def compute_centers(self, find_center):
        mask = self.mask
        mask = (mask * 255).astype(np.uint8)
        self._tissues = find_center(mask)
    
    def _get_components(self, mask):
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
    
    def get_centers_from_metadata(self, metadata, label):
        annotations = metadata[metadata['Classification'] == label]
        result = []
        for _, row in annotations.iterrows():
            bbox_mask = np.zeros(self.mask.shape, dtype=np.uint8)
            x  = int(round(row["Bounding_X"] / DOWNSAMPLE_FACTOR))
            y  = int(round(row["Bounding_Y"] / DOWNSAMPLE_FACTOR))
            w  = int(round(row["Bounding_Width"] / DOWNSAMPLE_FACTOR))
            h  = int(round(row["Bounding_Height"] / DOWNSAMPLE_FACTOR))
            bbox_mask[y : y + h, x : x + w] = 1
            tissue_mask = self.mask & bbox_mask 
            tissue_mask = (tissue_mask > 0).astype(np.uint8) * 255

            if not np.any(tissue_mask):
                continue
            else:
                cx = row['Centroid_X'] / DOWNSAMPLE_FACTOR
                cy = row['Centroid_Y'] / DOWNSAMPLE_FACTOR

                result.append({
                    'center': (cx, cy),
                    'mask': tissue_mask,
                })

        self._tissues = result

    def compute_peripheral_tumors_distance_to_center(self, bed_center):
        if self.is_tumoral:
            for t in self._tissues:
                dist_centers = np.array([distance.euclidean(bed_center, pxl_coord) for pxl_coord in t['peripherical-coord']])
                furthest_tum_pxl = np.argmax(dist_centers)
                t.update({'distance-center': (dist_centers[furthest_tum_pxl] * PIXEL_SIZE) / 1e3})
                t['peripherical-coord'] = t['peripherical-coord'][furthest_tum_pxl]

    def compute_distance_to_bed_border(self, bed_mask):
        mask = (bed_mask*255).astype(np.uint8)

        dist_map = distance_transform_edt(mask)
        if dist_map is not None:
            self._border_dist_map = dist_map
        else:
            raise Exception('Edt map is empty.')

        for t in self._tissues:
            dist_to_border = np.where(t['mask'] > 0, dist_map, np.inf)
            #min_coord = np.unravel_index(np.argmin(dist_to_border, axis=None), dist_to_border.shape)
            min_dist = np.min(dist_to_border)
            all_min_pxls = np.argwhere(dist_to_border == min_dist)
            t.update({
                'distance-border': ((min_dist-1) * PIXEL_SIZE) / 1e3,
                'peripherical-coord': all_min_pxls,
            })

    def compute_area_percentage(self, bed_mask):
        total_bed_pixels = np.count_nonzero(bed_mask)
        for t in self._tissues:
            t_pixels = np.count_nonzero(t['mask'])
            t.update({
                'area': t_pixels * (PIXEL_SIZE/1e3)**2,
                'area-percentage': t_pixels / total_bed_pixels
            })


    def get_histogram_pxl(self, bed_mask, ref="border", bed_center=None):
        tissues_pxl_coord = np.argwhere(self.mask)
        nb_pxl = len(tissues_pxl_coord)
        if ref == 'border':
            if self._border_dist_map is not None:
                dist_map = self._border_dist_map
            else:
                dist_map = distance_transform_edt(bed_mask)
                if dist_map is None:
                    return None

            dist_map_filtered = (dist_map[self.mask.astype(bool)] - 1) * (PIXEL_SIZE/1e3)
            hist, bins = np.histogram(dist_map_filtered, bins='scott')
            return hist/nb_pxl, bins

        elif ref == 'center':
            assert bed_center is not None, "bed_center should not be None."
            dist_to_bed_center = [distance.euclidean(bed_center, coord) * (PIXEL_SIZE/1e3) for coord in tissues_pxl_coord]
            hist, bins = np.histogram(dist_to_bed_center, bins='scott')
            return hist/nb_pxl, bins

        else:
            raise NotImplementedError(
                    f"""unknown reference '{ref}', expected one of:
                    border, center."""
                )

    def compute_normalize_distance(self, bed_mask, bed_center, pixels_coord, local_ratio = True):
        bed_contour, _ = cv2.findContours(bed_mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Squeeze contour to (N, 2) array [X, Y]
        b_coords = bed_contour[0].squeeze()

        cx, cy = bed_center[1], bed_center[0]

        # 2. Convert Boundary to Polar Coordinates
        b_dx = b_coords[:, 0] - cx
        b_dy = b_coords[:, 1] - cy
        
        b_theta = np.arctan2(b_dy, b_dx) # Range: [-pi, pi]
        b_r = np.hypot(b_dx, b_dy)       # Distance to center

        p_x = pixels_coord[:, 1]
        p_y = pixels_coord[:, 0]

        p_dx = p_x - cx
        p_dy = p_y - cy

        p_theta = np.arctan2(p_dy, p_dx)
        p_dist = np.hypot(p_dx, p_dy)

        if local_ratio:
            # 3. Handle C-Shapes: Sort by angle, then by distance
            # If a ray crosses the boundary twice, we want the *closest* intersection (smallest r)
            sort_idx = np.lexsort((b_r, b_theta)) 
            b_theta_sorted = b_theta[sort_idx]
            b_r_sorted = b_r[sort_idx]

            # Keep only unique angles (keeps the smallest 'r' due to lexsort) [for execution speed]
            unique_mask = np.append([True], b_theta_sorted[1:] != b_theta_sorted[:-1])
            theta_unique = b_theta_sorted[unique_mask]
            r_unique = b_r_sorted[unique_mask]

            # Handle the cyclic wrap-around (so pixels near -pi and pi interpolate correctly)
            theta_unique = np.concatenate(([theta_unique[0] - 2*np.pi], theta_unique, [theta_unique[-1] + 2*np.pi]))
            r_unique = np.concatenate(([r_unique[0]], r_unique, [r_unique[-1]]))
        
            # 5. Instantly calculate intersection distances for all 9 million pixels
            # This interpolates the boundary distance based on the pixel's angle
            r_boundary = np.interp(p_theta, theta_unique, r_unique)

            # 6. Calculate the final ratio
            # Prevent division by zero if the center is directly on a boundary edge
            r_boundary = np.where(r_boundary == 0, 1e-9, r_boundary)
        else:
            r_boundary = np.max(b_r)
        
        ratio = p_dist / r_boundary

        # Force the exact center pixel to be exactly 0.0
        ratio[p_dist == 0] = 0.0

        return ratio
    
    def _compute_distribution(self, kind, distances):
        nb_pxl = len(distances)
        if kind == 'hist':
            hist, bins = np.histogram(distances, bins='scott')
            return hist/nb_pxl, bins
        elif kind == 'kde':
            kde = stats.gaussian_kde(distances.flatten(), bw_method='scott')
            return kde
        else:
            raise NotImplementedError(
                f"""unknown kind '{kind}', expected one of:
                hist, kde."""
            )

    def get_distribution(self, bed_mask, ref="border", bed_center=None, kind='kde', norm_dist=True):
        tissues_pxl_coord = np.argwhere(self.mask)

        if ref == 'border':
            if self._border_dist_map is not None:
                dist_map = self._border_dist_map
            else:
                dist_map = distance_transform_edt(bed_mask)
                if dist_map is None:
                    return None

            distances = (dist_map[self.mask.astype(bool)] - 1) * (PIXEL_SIZE/1e3)

        elif ref == 'center':
            assert bed_center is not None, "bed_center should not be None."
            if norm_dist:
                dist_to_bed_center = self.compute_normalize_distance(bed_mask, bed_center,
                                                                      tissues_pxl_coord)
            else:
                dist_to_bed_center = [distance.euclidean(bed_center, coord) * (PIXEL_SIZE/1e3)
                                       for coord in tissues_pxl_coord]

            distances = dist_to_bed_center

        else:
            raise NotImplementedError(
                    f"""unknown reference '{ref}', expected one of:
                    border, center."""
                )

        return self._compute_distribution(kind, distances)

    def get_histogram_obj(self, ref="border", nb_bin=5):
        nb_obj = len(self._tissues)
        if ref == 'border':
            dist_to_border = [t['distance-border'] for t in self._tissues]
            hist, bins = np.histogram(dist_to_border, bins=nb_bin)
            return hist/nb_obj, bins
        elif ref == 'center':
            dist_to_bed_center = [t['distance-center'] for t in self._tissues]
            hist, bins = np.histogram(dist_to_bed_center, bins=nb_bin)
            return hist/nb_obj, bins
        else:
            raise NotImplementedError(
                    f"""unknown reference '{ref}', expected one of:
                    border, center."""
                )

    def get_objects(self):
        return self._tissues

    def get_top_k(self, k=3, mode="close", ref="border"):
        if ref in ['border', 'center']:
            dist_list = [t[f'distance-{ref}'] for t in self._tissues]
            sorted_indices = np.argsort(dist_list)
            if mode == 'close':
                top_k_idx = sorted_indices[:k]
            elif mode == 'far':
                top_k_idx = sorted_indices[-k:]
            else:
                raise NotImplementedError(
                f"""unknown mode '{mode}', expected one of 'close', 'far'."""
                )
        else:
            raise NotImplementedError(
                f"""unknown reference '{ref}', expected one of 'border', 'center'."""
            )
        return [t for idx, t in enumerate(self._tissues) if idx in top_k_idx]

    def __iter__(self):
        return iter(self._tissues)

    def __len__(self):
        return len(self._tissues)

    def __str__(self) -> str:
        str_repr = f"{self.name}\n"
        for idx, tissue in enumerate(self._tissues):
            str_repr += "-"*50 + "\n"
            str_repr += textwrap.dedent(f'''
                    Index: {idx}
                    Center: {tissue['center']}
                    Area percentage: {tissue['area']}
                    Distance-center (mm): {tissue['distance-center']}
                    Distance-border (mm): {tissue['distance-border']}
                    ''').strip()
            str_repr += "\n"
        return str_repr
