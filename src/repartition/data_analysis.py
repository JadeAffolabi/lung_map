import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist

from src.repartition.geometry_funcs import find_center_barycentre
from src.repartition.TissueCollection import *

def id_patient(slide_name: str):
    """
        Extract patient id from slide name.

    Parameters
    ----------
    slide_name : str

    Returns
    -------
        id : str
    """
    if 'BC' in slide_name:
        return slide_name.split(".")[1]
    elif 'DS' in slide_name:
        return "_".join(slide_name.split("_")[:2])
    return slide_name.split("_")[0]

def add_distance_info(data, tissues, tissue_type, slide_name) -> list[dict]:
    """
        Add informations about the tissues to the data list.

    Parameters
    ----------
    data : list[dict]
        data list for all the slides.
    tissues : TissueCollection
        tissue collection for one slide.
    tissue_type : str
    slide_name : str

    Returns
    -------
        data : list[dict]
    """
    for i, obj in enumerate(tissues.get_objects()):
        data.append({
                    'id-patient': id_patient(slide_name),
                    'id-slide': slide_name,
                    'type': tissue_type,
                    'component-id': i,
                    'dist-border': obj['distance-border'],
                    'dist-center': obj['distance-center'] if tissues.is_tumoral else None,
                    'min-dist-center': obj['min-dist-center'],
                    'norm-dist-border': obj['norm-dist-bord'],
                    'area': obj['area'],
                    'area-percentage': obj['area-percentage'],
                    'periph-coord': obj['peripherical-coord']
        })

    return data

def get_distribution(
        tissues: TissueCollection, masks: dict, tumor_bed: dict,
        tissue_type='tumor', return_kde=False,
        return_hist=False, local_ratio=True
) -> dict:
    """
        Compute the distribution of the tissues pixels.

    Parameters
    ----------
    tissues : TissueCollection
        tissue collection of one type for one slide.
    masks : dict
        masks for all tissue type (tumor, necrosis, stroma)
    tumor_bed : dict
        tumor bed informations (center and mask)
    tissue_type : str
    return_kde : bool
    return_hist : bool
    local_ratio : bool
        if true the normalized distance is the "distance relative locale" ,
        else it is  the "distance relative globale".

    Returns
    -------
        result : dict
    """
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
        for t, m in masks.items():
            norm_dist_to_center = compute_normalize_distance(
                    tumor_bed['mask'], tumor_bed['center'],
                    np.argwhere(m), local_ratio=local_ratio,
            )
            all_tissues_dist.update({
                t: norm_dist_to_center
            })

        result.update({
            'hist-proportion': get_proportion_hist(all_tissues_dist, tissue_type)
        })

    return result

def get_tissues_data(
        slides_annotation: dict, func_find_center, tissue_type='tumor',
        return_kde=False, return_hist=False, local_ratio=True
) -> tuple[pd.DataFrame, dict]:
    """
        Extract from slides annotations information such as distances and distribution of pixels
        for a specific class (tumor, necrosis, stroma)

    Parameters
    ----------
    slides_annotation : dict[str, dict]
    func_find_center : callable
        function to estimate the center of the tumor bed.
    tissue_type : str
        one of tumor, necrosis, stroma.
    return_kde : bool
    return_hist : bool
    local_ratio : bool
        if true the normalized distance is the "distance relative locale" ,
        else it is  the "distance relative globale".

    Returns
    -------
        distance_data, distribution_data : tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray] | stats.gaussian_kde]
    """
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
            except IndexError:
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

    distance_data = pd.DataFrame(distance_data)
    distance_data_final = distance_data.loc[get_furthest_tumors(distance_data, by='id-slide')]
    distance_data_final['min-dist-center'] = distance_data.loc[
        get_closest_tumors(distance_data, by='id-slide')
    ]['min-dist-center'].values
    distance_data_final['area-percentage'] = get_total_area_pct(distance_data, by='id-slide')

    distance_data_final.reset_index(inplace=True)
    distance_data_final.drop(columns=['index', 'type', 'component-id', 'area'], inplace=True)


    return distance_data_final, distribution_data

def get_furthest_tumors(data, by='id-slide'):
    """
        Returns the indices of the tumors components which are furthest from the center of the tumor bed.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing the informations (distances, etc.) for each tumor component for every slide.
    by : str
        column name to group by.

    Returns
    -------
        indices : pd.Series
    """
    return data.groupby(by)['dist-center'].idxmax()

def get_closest_tumors(data, by='id-slide'):
    """
        Returns the indices of the tumors components which are closest to the center of the tumor bed.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing the informations (distances, etc.) for each tumor component for every slide.
    by : str
        column name to group by.

    Returns
    -------
        indices : pd.Series
    """
    return data.groupby(by)['min-dist-center'].idxmin()

def get_total_area_pct(data, by='id-slide'):
    """
        Returns the sum of the area percentage for each element of the group specify by the "by" argument.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing the informations (distances, etc.) for each tumor component for every slide.
    by : str
        column name to group by.

    Returns
    -------
        values : pd.Series
    """
    return data.groupby(by)["area-percentage"].sum().values

def ratio_border_tumor(masks: dict[str, np.ndarray]) -> float:
    """
        Compute the proportion of border occupied by tumors.

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        set of mask for each class (tumor, necrosis, stroma, tumor_bed)

    Returns
    -------
        ratio_border_tumor : float
    """
    kernel = np.ones((3, 3), np.uint8)
    contour_bed = cv2.morphologyEx(masks['tumor_bed'].astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    contour_tumor = cv2.morphologyEx(masks['tumor'].astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

    tolerance_kernel = np.ones((5, 5), np.uint8) 
    contour_bed_tol = cv2.dilate(contour_bed, tolerance_kernel)

    common = cv2.bitwise_and(contour_bed_tol, contour_tumor)
    return np.count_nonzero(common) / np.count_nonzero(contour_bed)

def get_proportion_hist(all_tissues_dist: dict, tissue_type='tumor'):
    """
        Compute a histogram of the tissue proportion with respect to the normalized distance.
        ex: at midway (0.5) between center and border, there is 50% of tumor and 50% of others tissues.

    Parameters
    ----------
    all_tissues_dist : dict
        set of pixel distances for all tissue types (tumor, necrosis, stroma)
    tissue_type

    Returns
    -------
        proportion_hist, bins : tuple[np.ndarray, np.ndarray]
    """
    bins = np.linspace(0, 1, 20)
    count_per_bin = {
        t: np.histogram(all_tissues_dist[t], bins=bins)[0]
        for t in all_tissues_dist.keys()
    }
    sum_per_bin = np.zeros(len(bins)-1)
    for t in all_tissues_dist.keys():
        sum_per_bin += count_per_bin[t]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning
        )
        proportion_per_bin = count_per_bin[tissue_type] / sum_per_bin

    proportion_per_bin = np.nan_to_num(proportion_per_bin)
    return proportion_per_bin, bins

def get_distrib_model(
        model,
        distribution_data=None, thresholds=None, slide_annotations=None,
        tissue_type='tumor', func_find_center= find_center_barycentre,
    ):
    """
        Return for each slide a vector of characteristics based on the distribution model chose.

    Parameters
    ----------
    model : str
        model type
    distribution_data : dict
        distribution data for each slide
    thresholds : list[float]
        different thresholds for the model "vector of presence".
        ex: [0.2, 0.6, 0.8, 1]
    slide_annotations : dict[str, dict]
        slide annotations
    tissue_type : str
    func_find_center : callable
        function to estimate the center.

    Returns
    -------
        distrib : np.ndarray
    """
    distrib_list = []
    eval_points = np.linspace(0, 1, 20)

    if model in ['kde', 'hist']:
        if distribution_data is not None:
            for distrib in distribution_data.values():
                if model == 'kde':
                    kde_vec = distrib['kde']['center'](eval_points)
                    distrib_list.append(kde_vec / kde_vec.sum())
                else:
                    hist = distrib['hist-proportion'][0]
                    distrib_list.append(hist / hist.sum())
                    assert np.all(distrib['hist-proportion'][1] == eval_points), "Number of bins mismatch."
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
                    except IndexError:
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
    """
        Extract some features from the slide informations.
        Namely, the proportion of border occupied by tumors,
        the smallest distance of tumors to the center and the distance to the border of the furthest tumor component.
    Parameters
    ----------
    slides_annot : dict
        slide annotations
    distance_data : pd.DataFrame
        dataframe containing the informations (distances, etc.) for each tumor component for every slide.

    Returns
    -------
        features : np.ndarray
    """
    tumor_border_contact = [
        [ratio_border_tumor(masks)]
        for masks in slides_annot.values() if np.any(masks['tumor'])
    ]

    min_dist_center = [
        [distance_data[(distance_data['id-slide']==id_sld)]['min-dist-center'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    norm_dist_border = [
        [distance_data[(distance_data['id-slide']==id_sld)]['norm-dist-bord'].values[0]]
        for id_sld, masks in slides_annot.items() if np.any(masks['tumor'])
    ]

    tumor_border_contact = np.array(tumor_border_contact)
    min_dist_center = np.array(min_dist_center)
    norm_dist_border = np.array(norm_dist_border)

    feats = np.hstack(
        (
            tumor_border_contact,
            min_dist_center,
            norm_dist_border,
        )
    )

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    return feats


def compute_distance_matrix( 
        feats : np.ndarray,
        distrib: np.ndarray = None,
        distrib_type="kde",
    ) -> np.ndarray:
    """
        Calculate the distance or similarity between two slides based on their features.

    Parameters
    ----------
    feats : np.ndarray, shape=(n, k)
        The list of k features for n slides.
    distrib : np.ndarray, shape=(n, p), optional
        The list of distribution vector for n slides. Each distribution vector has p elements.
    distrib_type : str, optional
        distribution type if distrib is given.

    Returns
    -------
        dist_matrix : np.ndarray, shape=(n, n)
    """
    n = feats.shape[0] 
    dist_matrix = np.zeros((n, n))

    eval_points = np.linspace(0, 1, 20)

    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = abs(feats[i][0] - feats[j][0])**2 + np.linalg.norm(feats[i][1:] - feats[j][1:])
                            
            if (distrib is not None) and (distrib_type != "vect"):
                if distrib_type == "kde":
                    dist_matrix[i, j] += wasserstein_distance(
                        eval_points, eval_points,
                        distrib[i], distrib[j]
                    )
                elif distrib_type == "hist":
                    bin_mids    = (eval_points[:-1] + eval_points[1:]) / 2
                    dist_matrix[i, j] += wasserstein_distance(
                                            bin_mids, bin_mids,
                                            distrib[i], distrib[j]
                                        )

    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[j, i] = dist_matrix[i, j]

    return dist_matrix

def get_tumor_slides(slides_annotation: dict) -> np.ndarray[tuple[str, np.ndarray]]:
    """
        Returns only the slides with annotated tumors.

    Parameters
    ----------
    slides_annotation : dict

    Returns
    -------
        np.ndarray[[tuple[str, np.ndarray]]]
    """
    return np.array([(id_sld, masks) for id_sld, masks in slides_annotation.items()
            if np.any(masks['tumor'])])

def assign_cluster(labels_tum: np.ndarray, slides_annot: dict, slides_tumor: list) -> pd.DataFrame:
    """
        Assign clusters to slides.

    Parameters
    ----------
    labels_tum : np.ndarray
        predicted labels for slides with tumors.
    slides_annot : dict[str, dict]
        all annotated slides.
    slides_tumor : list[tuple[str, np.ndarray]]
        slides with tumors.
    Returns
    -------
        pd.DataFrame
    """
    df = pd.DataFrame(
        [sld_name for sld_name in slides_annot.keys()],
        columns=['id-slide']
    )
    df['cluster'] = 0
    for i, sld in enumerate(slides_tumor):
        id_sld = id_patient(sld[0])
        df.loc[df['id-slide']==id_sld, 'cluster'] = labels_tum[i] + 1
    return df

def compute_tumor_dispersion(tumor_mask: np.ndarray, bed_mask: np.ndarray) -> float | None:
    """
        Compute the dispersion score.
        See the "carnet expérimentation" for more details.

    Parameters
    ----------
    tumor_mask : np.ndarray
    bed_mask : np.ndarray

    Returns
    -------
        float
    """
    bed_contour, _ = cv2.findContours(
                bed_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bed_contour = bed_contour[0].squeeze()

    tum_points = cv2.findNonZero(tumor_mask)
    tum_points = np.expand_dims(tum_points, 1)
    tum_hull = cv2.convexHull(tum_points).squeeze()

    l_tum = get_diameter(tum_hull)
    l_bed = get_diameter(bed_contour)
    

    num_labels, _ = cv2.connectedComponents(tumor_mask, connectivity=8)
    num_components = num_labels - 1

    if num_components > 1 : 
        hull_mask = np.zeros_like(tumor_mask)
        cv2.drawContours(hull_mask, [tum_hull], -1, 255, thickness=-1)
        empty_mask = (hull_mask & ~tumor_mask) & bed_mask
        empty_area = np.count_nonzero(empty_mask)

    else:
        empty_area = 0

    tumor_area = np.count_nonzero(tumor_mask)
    bed_area = np.count_nonzero(bed_mask)

    m = min(1.0, l_tum/l_bed) + min(1.0, tumor_area/bed_area)**2 + empty_area/bed_area

    return m

def get_diameter(contour: np.ndarray) -> float:
    """
        Compute the pairwise distance between all contour points and return the maximum distance.

    Parameters
    ----------
    contour : np.ndarray
        contour points.

    Returns
    -------
        float
    """
    distances = pdist(contour, metric='euclidean')
    
    return np.max(distances)
