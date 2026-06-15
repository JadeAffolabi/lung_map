
import cv2
import numpy as np
from scipy.ndimage import find_objects
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely import get_parts
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

    def compute_normalize_distance_old(self, bed_mask, bed_center, pixels_coord, local_ratio = True):
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

            # Pour les pixels dans le masque, r_boundary ne peut pas être < p_dist
            r_boundary = np.maximum(r_boundary, p_dist)


            ratio = p_dist / r_boundary
        else:
            r_boundary = np.max(b_r)
            ratio = p_dist / r_boundary

        ratio[p_dist == 0] = 0.0

        """ problematic = ratio > 1

        if problematic.any():
            p_theta_prob = p_theta[problematic]
            p_dist_prob  = p_dist[problematic]

            angle_tol = np.deg2rad(0.5)  # tolérance angulaire
            r_adaptive = np.zeros(problematic.sum())

            for i, (theta_i, dist_i) in enumerate(zip(p_theta_prob, p_dist_prob)):
                # Tous les points du contour proches de cet angle
                diff = np.abs(b_theta_sorted - theta_i)
                diff = np.minimum(diff, 2*np.pi - diff)
                nearby = b_r_sorted[diff < angle_tol]

                if len(nearby) > 0:
                    nearby_sorted = np.sort(nearby)
                    # Premier r >= p_dist
                    idx = np.searchsorted(nearby_sorted, dist_i)
                    if idx < len(nearby_sorted):
                        r_adaptive[i] = nearby_sorted[idx]
                    else:
                        r_adaptive[i] = nearby_sorted[-1]  # vraiment hors du masque

            ratio[problematic] = p_dist_prob / np.where(r_adaptive == 0, 1e-9, r_adaptive) """
    
        return ratio
    
    def _raycast_r_boundary(self, cx, cy, p_dx_i, p_dy_i, p_dist_i, contour_line, r_max):
        norm = np.hypot(p_dx_i, p_dy_i)
        if norm == 0:
            return None
        dx_n, dy_n = p_dx_i / norm, p_dy_i / norm

        far_x = cx + dx_n * r_max * 2
        far_y = cy + dy_n * r_max * 2

        ray = LineString([(cx, cy), (far_x, far_y)])
        intersection = ray.intersection(contour_line)

        if intersection.is_empty:
            return None

        # Extraire tous les points d'intersection quel que soit le type
        pts = []
        for geom in get_parts(intersection):
            if geom.geom_type == 'Point':
                pts.append((geom.x, geom.y))
            elif hasattr(geom, 'coords'):
                pts.extend(geom.coords)

        if not pts:
            return None

        dists = [np.hypot(x - cx, y - cy) for x, y in pts]
        valid = [d for d in dists if d >= p_dist_i - 1e-6]

        return min(valid) if valid else None
    
    def _raycast_all(self, cx, cy, p_dx_prob, p_dy_prob, p_dist_prob, b_coords):
        """
        Ray casting vectorisé sur les segments du contour.
        Pour chaque pixel problématique, trouve la première intersection
        du rayon (centre → pixel) avec le contour, au-delà du pixel.
        """
        # Segments du contour : A → B
        A = b_coords[:-1]                          # (N, 2)
        B = b_coords[1:]                           # (N, 2)
        # Fermer le contour
        A = np.vstack([A, b_coords[-1:]])
        B = np.vstack([B, b_coords[:1]])
        AB = B - A                                 # vecteurs de segments (N, 2)

        n_prob = len(p_dx_prob)
        r_corrected = np.full(n_prob, np.nan)

        for i in range(n_prob):
            dx, dy = p_dx_prob[i], p_dy_prob[i]
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            # Direction unitaire du rayon
            dx_n, dy_n = dx / norm, dy / norm

            # Vecteur centre → A pour chaque segment
            OA = A - np.array([cx, cy])            # (N, 2)

            # Intersection rayon/segment par la règle de Cramer :
            # t = distance le long du rayon jusqu'à l'intersection
            # s = position sur le segment [0, 1]
            denom = dx_n * AB[:, 1] - dy_n * AB[:, 0]   # (N,)

            valid_denom = np.abs(denom) > 1e-10
            t = np.full(len(A), np.inf)
            s = np.full(len(A), np.inf)

            t[valid_denom] = (OA[valid_denom, 0] * AB[valid_denom, 1]
                            - OA[valid_denom, 1] * AB[valid_denom, 0]) / denom[valid_denom]
            s[valid_denom] = (OA[valid_denom, 0] * dy_n
                            - OA[valid_denom, 1] * dx_n) / denom[valid_denom]

            # Intersection valide : t >= p_dist (devant le pixel) et s ∈ [0, 1]
            valid = (t >= p_dist_prob[i] - 1e-6) & (s >= -1e-6) & (s <= 1 + 1e-6)

            if valid.any():
                r_corrected[i] = t[valid].min()

        return r_corrected


    def compute_normalize_distance(self, bed_mask, bed_center, pixels_coord, local_ratio=True):
        bed_contour, _ = cv2.findContours(
            bed_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        b_coords = bed_contour[0].squeeze()
        cx, cy = bed_center[1], bed_center[0]

        # Coordonnées polaires du contour
        b_dx = b_coords[:, 0] - cx
        b_dy = b_coords[:, 1] - cy
        b_theta = np.arctan2(b_dy, b_dx)
        b_r = np.hypot(b_dx, b_dy)

        # Coordonnées polaires des pixels
        p_x = pixels_coord[:, 1]
        p_y = pixels_coord[:, 0]
        p_dx = p_x - cx
        p_dy = p_y - cy
        p_theta = np.arctan2(p_dy, p_dx)
        p_dist = np.hypot(p_dx, p_dy)

        if local_ratio:
            # Tri par (theta, r) pour l'interpolation
            sort_idx = np.lexsort((b_r, b_theta))
            b_theta_sorted = b_theta[sort_idx]
            b_r_sorted = b_r[sort_idx]

            # Angles uniques + raccordement cyclique
            unique_mask = np.append([True], b_theta_sorted[1:] != b_theta_sorted[:-1])
            theta_unique = b_theta_sorted[unique_mask]
            r_unique = b_r_sorted[unique_mask]
            theta_unique = np.concatenate(([theta_unique[0] - 2*np.pi],
                                            theta_unique,
                                        [theta_unique[-1] + 2*np.pi]))
            r_unique = np.concatenate(([r_unique[0]], r_unique, [r_unique[-1]]))

            # Interpolation polaire
            r_boundary = np.interp(p_theta, theta_unique, r_unique)
            safe_r = np.where(r_boundary == 0, 1e-9, r_boundary)
            ratio = p_dist / safe_r

            # --- Correction par ray casting pour les pixels problématiques ---
            problematic = ratio > 1
            n_prob = problematic.sum()

            """ if n_prob > 0:
                print(f"Ray casting sur {n_prob} pixels problématiques...")

                # Contour comme polyligne Shapely (construit une seule fois)
                contour_line = LineString(
                    np.vstack([b_coords, b_coords[0]])  # ferme le contour
                )
                r_max = b_r.max()
                prob_idx = np.where(problematic)[0]

                corrected = 0
                for i in prob_idx:
                    r_true = self._raycast_r_boundary(
                        cx, cy,
                        p_dx[i], p_dy[i], p_dist[i],
                        contour_line, r_max
                    )
                    if r_true is not None:
                        ratio[i] = p_dist[i] / r_true
                        corrected += 1
                    # sinon on garde le ratio tel quel

                still_over = (ratio[prob_idx] > 1).sum()
                print(f"  Corrigés     : {corrected}")
                print(f"  Encore > 1   : {still_over}") """
            
            if problematic.any():
                prob_idx = np.where(problematic)[0]

                r_corrected = self._raycast_all(
                    cx, cy,
                    p_dx[problematic], p_dy[problematic], p_dist[problematic],
                    b_coords
                )

                # Appliquer les corrections trouvées
                found = ~np.isnan(r_corrected)
                ratio[prob_idx[found]] = p_dist[problematic][found] / r_corrected[found]

        else:
            r_max = np.max(b_r)
            ratio = p_dist / r_max

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

    def get_distribution(self, bed_mask, ref="border", bed_center=None,
                        kind='kde', norm_dist=True, local_ratio=True):
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
                                                                      tissues_pxl_coord, local_ratio)
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
