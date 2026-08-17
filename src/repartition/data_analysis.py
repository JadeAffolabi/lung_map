import math
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS, TSNE

from src.repartition.constants import CLASSES
from src.repartition.geometry_funcs import find_center_barycentre, find_center_skeleton
from src.repartition.TissueCollection import *

def id_patient(slide_name):
    if 'BC' in slide_name:
        return slide_name.split(".")[1]
    elif 'DS' in slide_name:
        return "_".join(slide_name.split("_")[:2])
    return slide_name.split("_")[0]

def add_distance_info(data, tissues, tissue_type, slide_name):
    for i, obj in enumerate(tissues.get_objects()):
        data.append({
                    'id-patient': id_patient(slide_name),
                    'id-slide': slide_name,
                    'type': tissue_type,
                    'distance-border': obj['distance-border'],
                    'distance-center': obj['distance-center'] if tissues.is_tumoral else None,
                    'min-dist-center': obj['min-dist-center'],
                    'area': obj['area'],
                    'area-percentage': obj['area-percentage'],
                    'periph-coord': obj['peripherical-coord']
        })

    return data

def get_distribution(tissues, masks, tumor_bed,
                     tissue_type='tumor', return_kde=False,
                     return_hist=False, local_ratio=True):
    result = dict()
    all_tissues_dist = dict()

    if return_kde:
        result.update({
            'kde': {
                'center': tissues.get_distribution(tumor_bed['mask'], 
                                                    ref='center', kind='kde',
                                                    bed_center=tumor_bed['center'],
                                                    norm_dist=True, 
                                                    local_ratio=local_ratio)}
        })

    if return_hist:
        for type, m in masks.items():
            norm_dist_to_center = compute_normalize_distance(
                    tumor_bed['mask'], tumor_bed['center'],
                    np.argwhere(m), local_ratio=local_ratio,
            )
            all_tissues_dist.update({
                type: norm_dist_to_center
            })

        result.update({
            'hist-proportion': get_type_proportion_hist(all_tissues_dist, tissue_type)
        })

    return result

def get_tissues_data(slides_annotation, func_find_center, tissue_type='tumor', return_kde=False, return_hist=False, local_ratio=True):
    distance_data = []
    distribution_data = {}
    for sld_name, masks in slides_annotation.items():
        if np.any(masks[tissue_type]):
            try:
                center_info = func_find_center(masks['tumor_bed'])
                if center_info is None:
                    print(f"Cannot compute center for slide {sld_name}")
                    continue
                else:
                    tumor_bed = {
                        'mask': masks['tumor_bed'],
                        'center': center_info[0]['center']
                    }
            except IndexError as e:
                continue

            tissues = TissueCollection(tissue_type, masks[tissue_type])
            tissues.compute_distance_to_bed_border(tumor_bed['mask'])
            tissues.compute_peripheral_tumors_distance_to_center(
                tumor_bed['mask'],
                tumor_bed['center']
            )
            tissues.compute_area_percentage(tumor_bed['mask'])

            distance_data = add_distance_info(
                distance_data, 
                tissues, 
                tissue_type, 
                sld_name
            )

            distribution_data[sld_name] = get_distribution(
                tissues, masks, tumor_bed, 
                tissue_type, return_kde, return_hist,
                local_ratio=local_ratio,
            )
        else:
            continue

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

    tolerance_kernel = np.ones((5, 5), np.uint8) 
    contour_bed_tol = cv2.dilate(contour_bed, tolerance_kernel)

    common = cv2.bitwise_and(contour_bed_tol, contour_tumor)
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

def f_x(x):
    return x**(2)

def g_x(x):
    return x**(1/2)

def get_distrib_model(
        model,
        distribution_data=None, thresholds=None, slide_annotations=None,
        tissue_type='tumor', func_find_center= find_center_barycentre,
    ):
    distrib_list = []
    eval_points = np.linspace(0, 1, 50)
    bins = np.linspace(0, 1, 20)

    res = []

    if model in ['kde', 'hist']:
        if distribution_data is not None:
            for distrib in distribution_data.values():
                if model == 'kde':
                    kde_vec = distrib['kde']['center'](eval_points)
                    distrib_list.append(kde_vec / kde_vec.sum())
                else:
                    hist = distrib['hist-proportion'][0]
                    distrib_list.append(hist / hist.sum())
                    assert np.all(distrib['hist-proportion'][1] == bins), "Number of bins mismatch."
        else:
            raise Exception(f"If model={model}, distribution_data must be given")
    elif model=='vect':
        if not isinstance(thresholds, (list, tuple)):
            raise TypeError(f"thresholds expected to be one of 'list' or 'tuple', got {type(thresholds)}")
        if thresholds is not None:
            for sld_name in slide_annotations.keys():
                masks = slide_annotations[sld_name]
                if np.any(masks[tissue_type]):
                    try:
                        bed_center = func_find_center(masks['tumor_bed'])[0]['center']
                    except IndexError as e:
                        continue
                    pixels_coord = np.argwhere(masks[tissue_type])

                    ratio = compute_normalize_distance(
                        masks['tumor_bed'],
                        bed_center,
                        pixels_coord,
                        local_ratio=True
                    )

                    neighborhood = []
                    for i, thresh in enumerate(thresholds):
                        in_neighb = np.logical_and(thresholds[i-1] < ratio, ratio <= thresh) \
                                        if i != 0 else ratio <= thresh
                        neighborhood.append(
                            1 if np.any(in_neighb) else 0
                        )

                    distrib_list.append(
                        neighborhood
                    )
        else:
            raise Exception(f"If model={model}, thresholds must be given")
    else:
        raise NotImplementedError(f"model={model} is not implemented.")

    return np.array(distrib_list)

def extract_features(slides_annot, distance_data):

    tumor_border_contact = [
        [ratio_border_tumor(masks)]
        for masks in slides_annot.values() if np.any(masks['tumor'])
    ]

    dist_center_closest_tum = get_closest_tumors(distance_data)
    min_dist_center = [
        [dist_center_closest_tum[(dist_center_closest_tum['id-slide']==id_sld)]['min-dist-center'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    furthest_tums = get_furthest_tumors(distance_data)
    min_dist_border = [
        [furthest_tums[(furthest_tums['id-slide']==id_sld)]['distance-border'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    tumor_border_contact = np.array(tumor_border_contact)
    min_dist_center = np.array(min_dist_center)
    min_dist_border = np.array(min_dist_border)

    feats = np.hstack(
        (
            tumor_border_contact,
            min_dist_center,
            min_dist_border, 
        )
    )

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    return feats

def compute_distance_matrix( 
        feats,
    ):
    n = feats.shape[0] 
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = abs(feats[i][0] - feats[j][0])**(2) + np.linalg.norm(feats[i][1:3] - feats[j][1:3])

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

def get_spread_dist(contours):
    l = contours[:, :, 0].argmin()
    r = contours[:, :, 0].argmax()
    t = contours[:, :, 1].argmin()
    b = contours[:, :, 1].argmax()

    dist_lr = distance.euclidean(contours[l][0], contours[r][0])
    dist_tb = distance.euclidean(contours[t][0], contours[b][0])

    return max(dist_lr, dist_tb)

def compute_tumor_spread(tumor_mask, bed_mask):

    bed_contour, _ = cv2.findContours(
                bed_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bed_contour = bed_contour[0].squeeze()

    tum_points = cv2.findNonZero(tumor_mask)
    tum_hull = cv2.convexHull(tum_points).squeeze()

    # bed_points = cv2.findNonZero(bed_mask)
    # bed_hull = cv2.convexHull(bed_points).squeeze()

    sp_tum = get_diameter(tum_hull)
    sp_bed = get_diameter(bed_contour)

    tumor_area = cv2.contourArea(tum_hull)
    bed_area = np.count_nonzero(bed_mask)

    m = min(1, sp_tum/sp_bed) + min(1, tumor_area / bed_area)

    return m

def get_diameter(contour):

    distances = pdist(contour, metric='euclidean')
    
    return np.max(distances)
