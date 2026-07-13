import math
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS, TSNE

from src.repartition.constants import CLASSES
from src.repartition.geometry_funcs import find_center_edt_barycentre
from src.repartition.TissueCollection import *

def id_patient(slide_name):
    if 'BC' in slide_name:
        return slide_name.split(".")[1]
    return slide_name.split("_")[0]

def add_distance_info(data, tissues, tissue_type, slide_name):
    for i, obj in enumerate(tissues.get_objects()):
        data.append({
                    'id-patient': id_patient(slide_name),
                    'id-slide': slide_name,
                    'type': tissue_type,
                    'distance-border': obj['distance-border'],
                    'distance-center': obj['distance-center'] if tissues.is_tumoral else None,
                    'max-dist-center': obj['max-dist-center'],
                    'min-dist-center': obj['min-dist-center'],
                    'area': obj['area'],
                    'area-percentage': obj['area-percentage'],
                    'periph-coord': obj['peripherical-coord']
        })

    return data

def get_distribution(sld_tissues, tumor_bed,
                     tissue_type='tumor', return_kde=False,
                     return_hist=False, local_ratio=True):
    result = dict()
    all_tissues_dist = dict()
    for type, tissues in sld_tissues.items():
        norm_dist_to_center = compute_normalize_distance(
                tumor_bed['mask'], tumor_bed['center'],
                np.argwhere(tissues.mask), local_ratio=local_ratio,
        )
        all_tissues_dist.update({
            type: norm_dist_to_center
        })
        if (type == tissue_type) and return_kde:
            result.update({
                'kde': {
                    'border': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='border', kind='kde',
                                                        local_ratio=local_ratio),
                    'center': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='center', kind='kde',
                                                        bed_center=tumor_bed['center'],
                                                        norm_dist=True, 
                                                        local_ratio=local_ratio)}
                })

        if (type == tissue_type) and return_hist:
            result.update({
                'hist': {
                    'border': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='border', kind='hist',
                                                        local_ratio=local_ratio),
                    'center': tissues.get_distribution(tumor_bed['mask'],
                                                        ref='center', kind='hist',
                                                        bed_center=tumor_bed['center'],
                                                        norm_dist=True,
                                                        local_ratio=local_ratio)}
            })

        if tissue_type in all_tissues_dist:
            result.update({
                'hist-proportion': get_type_proportion_hist(all_tissues_dist, tissue_type)
            })
    return result

def get_tissues_data(slides_annotation, return_kde=False, return_hist=False, local_ratio=True):
    distance_data = []
    distribution_data = {}
    for sld_name, masks in slides_annotation.items():
        try:
            tumor_bed = {
                'mask': masks['tumor_bed'],
                'center': find_center_edt_barycentre(masks['tumor_bed'])[0]['center']
            }
        except IndexError as e:
            continue
        sld_tissues = {}
        for tissue_type in CLASSES:
            if np.any(masks[tissue_type]):
                tissues = TissueCollection(tissue_type, masks[tissue_type])
                tissues.compute_distance_to_bed_border(tumor_bed['mask'])
                tissues.compute_peripheral_tumors_distance_to_center(
                    tumor_bed['mask'],
                    tumor_bed['center']
                )
                tissues.compute_area_percentage(tumor_bed['mask'])

                distance_data = add_distance_info(distance_data, tissues, tissue_type, sld_name)

                sld_tissues[tissue_type] = tissues

        if 'tumor' in sld_tissues:
            distribution_data[sld_name] = get_distribution(
                sld_tissues, tumor_bed, 
                'tumor', return_kde, return_hist,
                local_ratio=local_ratio,
            )

    return pd.DataFrame(distance_data), distribution_data


def compute_statistics(geo_data, tissue='tumor', by='id-slide'):
    grps = geo_data[geo_data['type'] == tissue].groupby(by)
    result = grps.agg(
        mean_dist_border=('distance-border', np.mean),
        std_dist_border=('distance-border', np.std),
        mean_dist_center=('distance-center', np.mean),
        std_dist_center=('distance-center', np.std),
        mean_area_percent=('area-percentage', np.mean),
        std_area_percent=('area-percentage', np.std),
        nb_tumor=('area', 'size')
    )
    return result

def get_areas(geo_data):
    result = geo_data.groupby([
        'id-patient', 
        'id-slide', 
        'type'
        ])[['area', 'area-percentage']].sum()
    result.reset_index(inplace=True)
    return result

def get_furthest_tumors(geo_data, by='id-slide'):
    tumor_geo = geo_data[geo_data['type']=='tumor']
    result = tumor_geo.loc[tumor_geo.groupby(by)['distance-center'].idxmax()]
    result.drop(columns=['periph-coord'], inplace=True)
    return result

def get_closest_tumors(geo_data, by='id-slide'):
    tumor_geo = geo_data[geo_data['type']=='tumor']
    result = tumor_geo.loc[tumor_geo.groupby(by)['min-dist-center'].idxmin()]
    result.drop(columns=['periph-coord'], inplace=True)
    return result

def ratio_border_tumor(masks):
    kernel = np.ones((3, 3), np.uint8)
    contour_bed = cv2.morphologyEx(masks['tumor_bed'].astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    contour_tumor = cv2.morphologyEx(masks['tumor'].astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    common = cv2.bitwise_and(contour_bed, contour_tumor)
    return np.count_nonzero(common) / np.count_nonzero(contour_bed)

def get_type_proportion_hist(all_tissues_dist, tissue_type='tumor'):
    bins = np.linspace(0, 1, 20)
    count_per_bin = {
        type: np.histogram(all_tissues_dist[type], bins=bins)[0]
        for type in all_tissues_dist.keys()
    }
    sum_per_bin = np.zeros(len(bins)-1)
    for type in all_tissues_dist.keys():
        sum_per_bin += count_per_bin[type]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning
        )
        proportion_per_bin = count_per_bin[tissue_type] / sum_per_bin

    proportion_per_bin = np.nan_to_num(proportion_per_bin)
    return proportion_per_bin, bins

def extract_features(distribution_data, eval_points, bins, distance_data, slides_annot):
    kde_list = []
    hist_prop_list = []
    other_feats = []
    for distrib in distribution_data.values():
        if 'kde' in distrib:
            kde_vec = distrib['kde']['center'](eval_points)
            kde_list.append(kde_vec / kde_vec.sum())
        
        hist = distrib['hist-proportion'][0]
        hist_prop_list.append(hist / hist.sum())
        assert np.all(distrib['hist-proportion'][1] == bins), "Number of bins mismatch."

        all_tissue_areas = get_areas(distance_data)

        tumor_border_contact = [ratio_border_tumor(masks) for masks in slides_annot.values() if np.any(masks['tumor'])]

        sld_tumor_percentage = [
            all_tissue_areas[(all_tissue_areas['id-slide']==id_sld) & (all_tissue_areas['type']=='tumor')]['area-percentage'].values[0]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
        ]

        dist_border_furthest_tum = get_furthest_tumors(distance_data)
        min_dist_border = [
            dist_border_furthest_tum[(dist_border_furthest_tum['id-slide']==id_sld)]['max-dist-center'].values[0]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
        ]

        dist_center_closest_tum = get_closest_tumors(distance_data)
        min_dist_center = [
            dist_center_closest_tum[(dist_center_closest_tum['id-slide']==id_sld)]['min-dist-center'].values[0]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
        ]

        other_feats = np.array([[contact**(1/4) + contact**(2), dist_border, dist_center] 
                                for contact, dist_border, dist_center in zip(
                                    tumor_border_contact,
                                    min_dist_border,
                                    min_dist_center, 
                                )])
        scaler = StandardScaler()
        other_feats = scaler.fit_transform(other_feats)

    return np.array(kde_list), np.array(hist_prop_list), np.array(other_feats)



def extract_features2(slides_annot, thresholds, distance_data, tissue_type='tumor'):
    tumor_border_percent = [
        [ratio_border_tumor(masks)] 
        for masks in slides_annot.values() if np.any(masks[tissue_type])
    ]

    all_tissue_areas = get_areas(distance_data)
    sld_tumor_percent = [
            [all_tissue_areas[(all_tissue_areas['id-slide']==id_sld) & (all_tissue_areas['type']=='tumor')]['area-percentage'].values[0]]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    dist_center_closest_tum = get_closest_tumors(distance_data)
    dist_center_furthest_tum = get_furthest_tumors(distance_data)
    min_dist_center = [
        [dist_center_closest_tum[(dist_center_closest_tum['id-slide']==id_sld)]['min-dist-center'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    max_dist_center = [
        [dist_center_furthest_tum[(dist_center_furthest_tum['id-slide']==id_sld)]['max-dist-center'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    center_neighborhood = []
    for sld_masks in slides_annot.values():
        if np.any(sld_masks[tissue_type]):
            try:
                bed_center = find_center_edt_barycentre(sld_masks['tumor_bed'])[0]['center']
            except IndexError as e:
                continue
            pixels_coord = np.argwhere(sld_masks[tissue_type])

            bed_contour, _ = cv2.findContours(
                sld_masks['tumor_bed'].astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            b_coords = bed_contour[0].squeeze()
            cx, cy = bed_center[1], bed_center[0]

            
            b_dx = b_coords[:, 0] - cx
            b_dy = b_coords[:, 1] - cy
            b_r = np.hypot(b_dx, b_dy)
            r_max = np.max(b_r)
    
            ratio = compute_normalize_distance(
                sld_masks['tumor_bed'],
                bed_center,
                pixels_coord,
                local_ratio=True
            )

            all_pixels_coord = np.argwhere(sld_masks['tumor_bed'])
            a_x = all_pixels_coord[:, 1]
            a_y = all_pixels_coord[:, 0]
            a_dx = a_x - cx
            a_dy = a_y - cy
            a_dist = np.hypot(a_dx, a_dy)
            all_ratio = a_dist / r_max

            neighborhood = []
            for i, r_thresh in enumerate(thresholds):
                in_neighb = np.logical_and(thresholds[i-1] < ratio, ratio <= r_thresh) \
                                if i != 0 else ratio <= r_thresh
                all_in_neighb = np.logical_and(thresholds[i-1] < all_ratio, all_ratio <= r_thresh) \
                                    if i != 0 else ratio <= r_thresh
                """ neighborhood.append(
                    np.sum(in_neighb) / np.sum(all_in_neighb)
                ) """
                neighborhood.append(
                    1 if np.any(in_neighb) else 0
                )

            center_neighborhood.append(
                neighborhood
            )
    
    center_neighborhood = np.array(center_neighborhood)
    tumor_border_percent = np.array(tumor_border_percent)
    sld_tumor_percent = np.array(sld_tumor_percent)
    min_dist_center = np.array(min_dist_center)
    max_dist_center = np.array(max_dist_center)

    def f_x(x):
        return x**(1/2) + x**(2)

    return np.hstack(
        (f_x(tumor_border_percent), f_x(min_dist_center), f_x(max_dist_center), center_neighborhood)
        #(tumor_border_percent, min_dist_center, max_dist_center)

    )


def get_center_neighbor_ratio(slides_annot, threshold, tissue_type='tumor'):
    
    center_neighborhood = []
    for sld_masks in slides_annot.values():
        if np.any(sld_masks[tissue_type]):
            try:
                bed_center = find_center_edt_barycentre(sld_masks['tumor_bed'])[0]['center']
            except IndexError as e:
                continue
            all_pixels_coord = np.argwhere(sld_masks['tumor_bed'])
            tissue_pixels_coord = np.argwhere(sld_masks[tissue_type])

            bed_contour, _ = cv2.findContours(
                sld_masks['tumor_bed'].astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            b_coords = bed_contour[0].squeeze()
            cx, cy = bed_center[1], bed_center[0]

            # Coordonnées polaires du contour
            b_dx = b_coords[:, 0] - cx
            b_dy = b_coords[:, 1] - cy
            b_r = np.hypot(b_dx, b_dy)

            p_x = tissue_pixels_coord[:, 1]
            p_y = tissue_pixels_coord[:, 0]
            p_dx = p_x - cx
            p_dy = p_y - cy
            p_dist = np.hypot(p_dx, p_dy)

            a_x = all_pixels_coord[:, 1]
            a_y = all_pixels_coord[:, 0]
            a_dx = a_x - cx
            a_dy = a_y - cy
            a_dist = np.hypot(a_dx, a_dy)

            r_max = np.max(b_r)
            tissue_ratio = p_dist / r_max
            all_ratio = a_dist / r_max

            tissue_in_neighb = tissue_ratio <= threshold
            all_in_neighb = all_ratio <= threshold
            neighb_proportion = np.sum(tissue_in_neighb) / np.sum(all_in_neighb)
            """ for r_thresh in thresholds:
                in_neighb = ratio <= r_thresh
                neighb_proportion.append(
                    np.sum(in_neighb) / len(pixels_coord)
                ) """
            center_neighborhood.append(
                neighb_proportion
            )
    
    center_neighborhood = np.array(center_neighborhood)

    return center_neighborhood

def mixed_distance(hist1, hist2, feat1, feat2, alpha=0.6):
    d_hist = wasserstein_distance(hist1, hist2)
    d_feat = np.linalg.norm(feat1 - feat2)
    return alpha * d_hist + (1 - alpha) * d_feat

def compute_histogram_dist(histograms, feats=None, mixed=False, alpha=0.6):
    n = len(histograms)  
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if mixed and (feats is not None):
                d = mixed_distance(histograms[i], histograms[j],
                                   feats[i], feats[j], alpha)
            else:
                d = wasserstein_distance(histograms[i], histograms[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d 
    return dist_matrix

def compute_distance_matrix(kde_arr, hist_arr, weights, eval_points, bins, additional_feats=None):
    n = len(kde_arr)  
    dist_matrix = np.zeros((n, n))
    dist_kde = np.zeros((n, n))
    dist_hist= np.zeros((n, n))
    dist_feat = np.zeros((n, n))
    bin_mids    = (bins[:-1] + bins[1:]) / 2 
    for i in range(n):
        for j in range(i+1, n):
            dist_kde[i, j] = wasserstein_distance(
               eval_points, eval_points,
               kde_arr[i], kde_arr[j]
            )
            dist_hist[i,j] = wasserstein_distance(
                bin_mids, bin_mids,
                #eval_points, eval_points,
                hist_arr[i], hist_arr[j]
            )
            if additional_feats is not None:
                dist_feat[i,j] = np.linalg.norm(
                    additional_feats[i] - additional_feats[j]
                )
    
    if additional_feats is not None:
        dist_matrix = weights[0]*dist_kde/dist_kde.max() + weights[1]*dist_hist/dist_hist.max() \
                    + weights[2]*dist_feat/dist_feat.max()
    else:
        dist_matrix = weights[0]*dist_kde/dist_kde.max() + weights[1]*dist_hist/dist_hist.max()
    
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

def compute_distance_matrix2(kde_arr, eval_points, 
                             tumor_border_contact,
                             vect,
    ):
    n = len(kde_arr)  
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = wasserstein_distance(
               eval_points, eval_points,
               kde_arr[i], kde_arr[j]
            ) + 1*abs(tumor_border_contact[i] - tumor_border_contact[j])**(2) \
              + 0*np.linalg.norm(vect[i][:3] - vect[j][:3], ord=2) \
              #+ np.linalg.norm(vect[i][3:] - vect[j][3:])**(1/2)
              #+ 0.0*abs(tumor_center_dist[i] - tumor_center_dist[j])**(1) \


    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

def get_tumor_slides(slides_annotation):
    return [(id, masks) for id, masks in slides_annotation.items() 
            if np.any(masks['tumor'])]

def get_slides_cluster(labels_tum, slides_annot, slides_tumor):
    df = pd.DataFrame(
        [id_patient(sld_name) for sld_name in slides_annot.keys()],
        columns=['id-patient']
    )
    df['cluster'] = 0
    for i, sld in enumerate(slides_tumor):
        id = id_patient(sld[0])
        df.loc[df['id-patient']==id, 'cluster'] = labels_tum[i] + 1
    return df