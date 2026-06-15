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
from src.repartition.TissueCollection import TissueCollection

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
                    'area': obj['area'],
                    'area-percentage': obj['area-percentage'],
                    'periph-coord': obj['peripherical-coord']
        })

    return data

def get_distribution(sld_tissues, tumor_bed,
                     tissue_type='tumor', return_kde=False, return_hist=False):
    result = dict()
    all_tissues_dist = dict()
    for type, tissues in sld_tissues.items():
        norm_dist_to_center = tissues.compute_normalize_distance(
                tumor_bed['mask'], tumor_bed['center'],
                np.argwhere(tissues.mask)
        )
        all_tissues_dist.update({
            type: norm_dist_to_center
        })
        if (type == tissue_type) and return_kde:
            result.update({
                'kde': {
                    'border': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='border', kind='kde'),
                    'center': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='center', kind='kde',
                                                        bed_center=tumor_bed['center'],
                                                        norm_dist=True)}
                })

        if (type == tissue_type) and return_hist:
            result.update({
                'hist': {
                    'border': tissues.get_distribution(tumor_bed['mask'], 
                                                        ref='border', kind='hist'),
                    'center': tissues.get_distribution(tumor_bed['mask'],
                                                        ref='center', kind='hist',
                                                        bed_center=tumor_bed['center'],
                                                        norm_dist=True)}
            })

        if tissue_type in all_tissues_dist:
            result.update({
                'hist-type-proportion': get_type_proportion_hist(all_tissues_dist, tissue_type)
            })
    return result

def get_tissues_data(slides_annotation, return_kde=False, return_hist=False):
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
                tissues.compute_peripheral_tumors_distance_to_center(tumor_bed['center'])
                tissues.compute_area_percentage(tumor_bed['mask'])

                distance_data = add_distance_info(distance_data, tissues, tissue_type, sld_name)

                sld_tissues[tissue_type] = tissues

        if 'tumor' in sld_tissues:
            distribution_data[sld_name] = get_distribution(
                sld_tissues, tumor_bed, 
                'tumor', return_kde, return_hist
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
    result = tumor_geo.loc[tumor_geo.groupby(by)['distance-center'].idxmin()]
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

def extract_features(distribution_data, eval_points, bins, distance_data=None, slides_annot=None, only_distrib=True):
    kde_list = []
    hist_prop_list = []
    other_feats = []
    for distrib in distribution_data.values():
        if 'kde' in distrib:
            kde_vec = distrib['kde']['center'](eval_points)
            kde_list.append(kde_vec / kde_vec.sum())
        
        hist = distrib['hist-type-proportion'][0]
        hist_prop_list.append(hist / hist.sum())
        assert np.all(distrib['hist-type-proportion'][1] == bins), "Number of bins mismatch."

    if not only_distrib:
        assert (distance_data is not None) and (slides_annot is not None), "'distance_data' and 'slides_annot' must be given."
        all_tissue_areas = get_areas(distance_data)

        tumor_border_contact = [ratio_border_tumor(masks) for masks in slides_annot.values() if np.any(masks['tumor'])]

        sld_tumor_percentage = [
            all_tissue_areas[(all_tissue_areas['id-slide']==id_sld) & (all_tissue_areas['type']=='tumor')]['area-percentage'].values[0]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
        ]

        min_dist_border = get_furthest_tumors(distance_data)
        tumor_border_dist = [
            min_dist_border[(min_dist_border['id-slide']==id_sld)]['distance-border'].values[0]
            for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
        ]

        other_feats = np.array([[contact, areas] 
                                for contact, areas in zip(
                                    tumor_border_contact, 
                                    sld_tumor_percentage, 
                                )])
        scaler = StandardScaler()
        other_feats = scaler.fit_transform(other_feats)

    return np.array(kde_list), np.array(hist_prop_list), other_feats

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
