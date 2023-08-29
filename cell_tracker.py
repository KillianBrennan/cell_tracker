#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
finds cells and tracks them
---------------------------------------------------------
IN
xarray dataset with one ensemble member
tracking parameters (thresholds)
---------------------------------------------------------
OUT
xarray dataset with tracks
---------------------------------------------------------
EXAMPLE CALLS
---------------------------------------------------------
TODOS

---------------------------------------------------------
Killian P. Brennan
16.05.2023
---------------------------------------------------------
"""

import sys

import copy
import os

import json

import numpy as np
import math
from tqdm import tqdm
import xarray as xr

from scipy import ndimage
from scipy import interpolate
from scipy.ndimage import rotate


import itertools
import collections

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.segmentation import expand_labels
from skimage.segmentation import flood
from skimage.morphology import h_maxima


def track_cells(
    fields,
    datelist,
    field_static,
    prominence=10,
    threshold=5,
    min_distance=6,
    fill_method="watershed",
    advection_method="movement_vector",
    dynamic_tracking=4,
    v_limit=10,
    min_area=16,
    alpha=0.5,
    velocity_score_weight=0.5,
    min_lifespan=30,
    aura=0,
    quiet=True,
    cluster_size_limit=16,
):
    """
    finds cells using initial cells and tracking them forwards through overlapping area from t-1 to t

    in
    fields: np.array of 2d meteorological field, dimension is (t, x, y)
    datelist: list of datetime objects
    field_static: dictionary with static fields (lat and lon), dimension is (x, y)
    prominence: prominence of local maxima to be considered a distinct cell, float
    threshold: threshold for minimum value considered as within cell, float
    min_distance: minimum distance (in gridpoints) between local maxima, float
    fill_method: method used to fill areas between cells, string
    advection_method: method used to advect cells, string
    dynamic_tracking: number of timesteps used for advecting search mask, int
    v_limit: limit of distance (in gridpoints) per timestept to advect search mask, float
    min_area: minimum area (in gridpoints) for a cell to be considered, int
    alpha: weight of overlap in score, float
    velocity_score_weight: weight of velocity in score, float
    min_lifespan: minimum lifespan (in minutes) for a cell to be considered, float
    aura: number of gridpoints to dilate labels, int
    quiet: suppress tqdm output, bool
    cluster_size_limit: maximum number of cells in a cluster, before more crude solution is applied to solving cluster, int

    out
    cells: list of cell objects, list
    """
    cells_alive = []  # list of active cell objects
    cells_dead = []  # list of deceased cell objects
    cells = []  # list of final cell objects
    cell_id = 0  # initial unique cell id
    last_above_threshold = None
    last_labeled = []
    flow_field = None
    prev_field = None

    # tracking loop
    if not quiet:
        fields_tqdm = tqdm(fields)
    else:
        fields_tqdm = fields

    for i, field in enumerate(fields_tqdm):
        nowdate = datelist[i]

        labeled, above_threshold = label_local_maximas(
            field,
            prominence,
            threshold,
            min_distance,
            fill_method,
            aura,
        )

        if prev_field is None:
            prev_field = field

        # assign lables at current timestep from areas overlapping to last timestep
        cells_alive, labels, cell_id = assign_new_labels(
            cells_alive,
            labeled,
            last_labeled,
            field,
            prev_field,
            cell_id,
            nowdate,
            flow_field,
            advection_method,
            dynamic_tracking,
            v_limit,
            min_area,
            alpha,
            velocity_score_weight,
            cluster_size_limit,
        )

        # initiate new cells from labels remaining in list and add to cells
        for label in labels:
            # logging.info(f"initiating new cell, id: {cell_id}")
            cells_alive.append(Cell(cell_id, label, nowdate))
            cells_alive[-1].area_gp.append(
                np.count_nonzero(labeled == cells_alive[-1].label[-1])
            )
            cell_id += 1

        # remove dead cells from cells_alive
        for cell in cells_alive:
            if not cell.alive:
                cells_dead.append(cell)
        cells_alive = [cell for cell in cells_alive if cell.alive]

        # todo: check should not be needed if code worked, can be deleted
        for cell in cells_alive:
            if not cell.alive:
                print("error: dead cell in alive list")
                sys.exit()

        # do math with all alive cells at timestep
        _ = [
            cell.cal_spatial(
                field_static,
                np.argwhere(labeled == cell.label[-1]),
                field[labeled == cell.label[-1]],
            )
            for cell in cells_alive
        ]

        _ = [cell.check_consistency() for cell in cells_alive]

        flow_field = generate_flow_field(cells_alive, field.shape, 10)

        last_labeled = labeled
        last_above_threshold = above_threshold

    # post processing
    cells = cells_alive + cells_dead

    cell_ids = []
    for cell in cells:
        cell_ids.append(cell.cell_id)

    cells = [cell for _, cell in sorted(zip(cell_ids, cells))]

    cell_ids = []
    for cell in cells:
        cell_ids.append(cell.cell_id)

    _ = [cell.post_processing() for cell in cells]

    # filter cells and append short-lived childs to parents (or merged_to)
    for cell in cells:
        if cell.lifespan < np.timedelta64(min_lifespan, "m"):
            if cell.parent:
                cells[cell_ids.index(cell.parent)].append_cell(cell, field_static)
            if cell.merged_to:
                cells[cell_ids.index(cell.merged_to)].append_cell(cell, field_static)

    cells = [
        cell for cell in cells if (cell.lifespan >= np.timedelta64(min_lifespan, "m"))
    ]

    cells = remap_ids(cells)

    for cell in cells:
        if cell.parent:
            cell.insert_split_timestep(cells[cell.parent])
        if cell.merged_to is not None:
            cell.insert_merged_timestep(cells[cell.merged_to])

    _ = [cell.post_processing() for cell in cells]
    start_dates = []
    for cell in cells_dead:
        start_dates.append(cell.datelist[0])
    start_dates.sort()

    # dataset = convert_tracks_to_netcdf(cells)

    if cells is not None:
        cells = fill_gaps(cells, fields, datelist)

    return cells


def fill_gaps(cells, fields, datelist):
    """
    fills gaps in swaths left between timesteps
    smudging of cells along track path

    in
    cells: list of cell objects, list
    fields: list of 2d meteorological fields, list
    datelist: list of datetime objects, list

    out
    cells: list of cell objects, list
    """
    # make 5x virtual supersampling
    supersampling = 5
    last_field = None
    for i, nowdate in enumerate(datelist):
        field = fields[i]
        for cell in cells:
            if nowdate in cell.datelist:
                index = cell.datelist.index(nowdate)
                if (index > 0) & (last_field is not None):
                    cell_coordinates = cell.field[index]
                    cell_values = field[cell_coordinates[:, 0], cell_coordinates[:, 1]]
                    cell_center = [cell.mass_center_x[index], cell.mass_center_y[index]]
                    last_cell_coordinates = cell.field[index - 1]
                    last_cell_values = last_field[
                        last_cell_coordinates[:, 0], last_cell_coordinates[:, 1]
                    ]
                    last_cell_center = [
                        cell.mass_center_x[index - 1],
                        cell.mass_center_y[index - 1],
                    ]

                    delta_x = cell.delta_x[index]
                    delta_y = cell.delta_y[index]
                    interpolated, min_coords, max_coords = interpolate_footprints(
                        cell_values,
                        last_cell_values,
                        cell_coordinates,
                        last_cell_coordinates,
                        delta_x,
                        delta_y,
                        supersampling,
                    )

                    if cell.swath is None:
                        cell.swath = np.zeros_like(field)

                    # check if interpolated shape is same as swath shape (fails for cells at the edge of the domain)
                    if (
                        cell.swath[
                            min_coords[0] : max_coords[0],
                            min_coords[1] : max_coords[1],
                        ].shape
                        == interpolated.shape
                    ):
                        cell.swath[
                            min_coords[0] : max_coords[0], min_coords[1] : max_coords[1]
                        ] = np.max(
                            np.dstack(
                                (
                                    cell.swath[
                                        min_coords[0] : max_coords[0],
                                        min_coords[1] : max_coords[1],
                                    ],
                                    interpolated,
                                )
                            ),
                            2,
                        )
                    else:
                        pass
        last_field = field

    return cells


def generate_flow_field(cells, grid_shape, subsampling=1):
    """
    generates flow fieldeld from sparse cell movement vectors

    in
    cells: list of cell objects, list
    grid_shape: shape of netcdf field, tuple
    subsampling: factor for spatial subsampling, int

    out
    flow_field: flow field with linear interpolation between active cells, array
    """
    x_pos = []
    y_pos = []
    u_vel = []
    v_vel = []
    grid_shape = tuple(math.ceil(dim / subsampling) for dim in grid_shape)

    for cell in cells:
        if len(cell.datelist) > 3:
            x_pos.append(cell.mass_center_x[-1])
            y_pos.append(cell.mass_center_y[-1])
            u_vel.append(np.mean(cell.delta_x[-3:-1]))
            v_vel.append(np.mean(cell.delta_y[-3:-1]))

    if len(u_vel) >= 5:  # interpolation needs at least 4 values
        positions = np.transpose(np.array([x_pos, y_pos]))
        grid_x, grid_y = np.mgrid[0 : grid_shape[0], 0 : grid_shape[1]]

        grid_u = interpolate.griddata(
            positions,
            u_vel,
            (grid_x, grid_y),
            method="linear",
            fill_value=np.mean(u_vel),
        )
        grid_v = interpolate.griddata(
            positions,
            v_vel,
            (grid_x, grid_y),
            method="linear",
            fill_value=np.mean(v_vel),
        )

        flow_field = np.zeros(grid_shape + (2,))

        flow_field[:, :, 0] = grid_u
        flow_field[:, :, 1] = grid_v

        # logging.info("linear interpolation used for flow field")

        return flow_field

    elif len(u_vel) > 0:  # simple average of all cell vectors for less than 4 cells
        # flow_field = np.zeros(cells[0].field[0].shape + (2,))
        flow_field = np.zeros((1, 1, 2))
        flow_field[:, :, 0] = np.mean(u_vel)
        flow_field[:, :, 1] = np.mean(v_vel)

        # logging.info("simple mean used for flow field")

        return flow_field

    else:
        return None


def label_local_maximas(field, prominence, threshold, min_distance, fill_method, aura):
    """
    labels areas of lokal peaks (separated by min_distance)

    in
    field: 2d meteorological field, array
    prominence: prominence of local maxima to be considered a distinct cell, float
    threshold: threshold for minimum value considered as within cell, float
    min_distance: minimum distance (in gridpoints) between local maxima, float
    fill_method: method used to fill areas between cells, string
    aura: number of gridpoints to dilate labels, int

    out
    labeled: connected component labeled with unique label, array
    above_threshold: above threshold binary area, array
    """

    field_shifted = field + prominence - threshold
    field_shifted[field < threshold] = 0

    mask_prominence = h_maxima(field_shifted, prominence)
    peaks_prominence = np.argwhere(mask_prominence)

    peaks_distance = peak_local_max(
        field, min_distance=min_distance, threshold_abs=threshold
    )

    peaks = peaks_distance[np.isin(peaks_distance, peaks_prominence).all(axis=1)]

    mask = np.zeros(field.shape, dtype=bool)
    mask[tuple(peaks.T)] = True

    if fill_method == "fill_peaks":
        # from each peak fill until peak_value/2 is reached
        above_threshold = np.zeros(field.shape, dtype=bool)
        for peak in peaks:
            peak_value = field[tuple(peak)]
            above_threshold = above_threshold + flood(
                field, tuple(peak.T), tolerance=peak_value / 2
            )
        # connected-component array of labels
        labeled, _ = ndimage.label(above_threshold)

    elif fill_method == "watershed":
        # use watershed method, fills adjacent cells so they touch at watershed line
        above_threshold = field > threshold

        markers, _ = ndimage.label(mask)
        labeled = watershed(-field, markers, mask=above_threshold)

    # dilate labels
    labeled = expand_labels(labeled, distance=aura)

    return labeled, above_threshold


def advect_array(flow_field, array, new_delta_x, new_delta_y):
    """
    advects cell labels according to movement vector
    todo: get rid of roll-over
    todo: change to simple addition of velocity*dt to cell coordinates

    in
    flow_field: estimated flow field, array
    array: 2d meteorological field, array
    new_delta_x: movement vector x-component, float
    new_delta_y: movement vector y-component, float

    out
    array: advected array, array
    """
    if new_delta_x is None:
        if flow_field is None:
            # no advection possible
            return array
        else:
            if flow_field.shape == (1, 1, 2):  # mean flow field
                new_delta_x = flow_field[0, 0, 0]
                new_delta_y = flow_field[0, 0, 1]
            else:
                coordinates = np.argwhere(array)
                center = np.mean(coordinates, axis=0)
                center = [int(round(x / 10)) for x in center]
                new_delta_x = flow_field[center[0], center[1], 0]
                new_delta_y = flow_field[center[0], center[1], 1]

    array = np.roll(array, int(round(new_delta_x)), axis=0)
    array = np.roll(array, int(round(new_delta_y)), axis=1)

    return array


def determine_cell_movement(delta_x, delta_y, n_timesteps_max, v_limit):
    """
    determineds cell movement for advection

    in
    delta_x: movement vector x-component, list
    delta_y: movement vector y-component, list
    n_timesteps_max: number of past timesteps used in cell advection
    v_limit: limit for array shifting, float

    out
    new_delta_x: movement vector x-component, float
    new_delta_y: movement vector y-component, float
    """
    if len(delta_x) < 2:  # first is always 0
        return None, None

    # at the beginning not all past timesteps are available
    n_timesteps = min(len(delta_x) - 1, n_timesteps_max)

    new_delta_x = 0
    new_delta_y = 0
    for i in range(n_timesteps):  # geometric sum with the nth last timesteps
        new_delta_x = (
            new_delta_x + 2**i / (2**n_timesteps - 1) * delta_x[-n_timesteps + i]
        )
        new_delta_y = (
            new_delta_y + 2**i / (2**n_timesteps - 1) * delta_y[-n_timesteps + i]
        )

    delta_xy = np.sqrt(new_delta_x**2 + new_delta_y**2)

    if delta_xy > v_limit:  # limit velocity
        new_delta_x = new_delta_x / delta_xy * v_limit
        new_delta_y = new_delta_y / delta_xy * v_limit

    return new_delta_x, new_delta_y


def assign_new_labels(
    cells,
    labeled,
    last_labeled,
    field,
    prev_field,
    cell_id,
    nowdate,
    flow_field,
    advection_method,
    dynamic_tracking,
    v_limit,
    min_area,
    alpha,
    velocity_score_weight,
    cluster_size_limit,
):
    """
    assigns new labeles to all alive cells and terminates dying ones

    in
    cells: list of cell objects, list
    labeled: labeled cell areas, array
    last_labeled: last labeled cell areas, array
    field: 2d meteorological field, array
    prev_field: previous 2d meteorological field, array
    cell_id: last used cell identifier, int
    nowdate: current datetime being investigated, datetime
    flow_field: estimate of flow field, array
    advection_method: method used to advect cells, string
    dynamic_tracking: number of timesteps used for advecting search mask, int
    v_limit: limit of distance (in gridpoints) per timestept to advect search mask, float
    min_area: minimum area (in gridpoints) for a cell to be considered, int
    alpha: weight of overlap in score, float
    velocity_score_weight: weight of velocity in score, float
    cluster_size_limit: maximum number of cells in a cluster, before more crude solution is applied to solving cluster, int

    out
    cells: list of cell objects, list
    labels: remaining labels that were not assigned, int
    cell_id: last used cell identifier, int
    """

    # logging.info("-------------------------------------")
    # logging.info("nowdate is " + nowdate.strftime("%m.%d %H:%M"))

    # will be pruned as labels are assigned
    labels = np.unique(labeled).tolist()
    if 0 in labels:
        labels.remove(0)  # not a cell, 0 is background

    child_cells = []  # needs to be stored intermittently for not to break cells loop

    (
        active_ids,
        candidates,
        counts,
        overlap,
        last_active_area,
        last_velocity,
        last_center,
    ) = find_overlaps(
        cells,
        labeled,
        last_labeled,
        field,
        prev_field,
        flow_field,
        advection_method,
        dynamic_tracking,
        v_limit,
        min_area,
    )

    available_labels = set(candidates)

    candidate_center = []
    for candidate in candidates:
        coordinates = np.argwhere(labeled == candidate)
        area_center = np.mean(coordinates, axis=0)
        max_pos = coordinates[np.argmax(field[labeled == candidate])]
        candidate_center.append(np.mean((area_center, max_pos), 0).tolist())

    # remove existing cells from initiation list (represents new cells that will be initiated later)
    labels = [x for x in labels if x not in available_labels]

    new_ids, new_labels, scores = find_correspondences(
        active_ids,
        candidates,
        counts,
        overlap,
        last_active_area,
        last_velocity,
        last_center,
        candidate_center,
        alpha,
        velocity_score_weight,
        cluster_size_limit,
    )

    splitting_ids = [
        id for id, count in collections.Counter(new_ids).items() if count > 1
    ]
    merging_labels = [
        lbl for lbl, count in collections.Counter(new_labels).items() if count > 1
    ]

    splitting_labels = []
    splitting_areas = []
    for id in splitting_ids:
        lbl = []
        area = []
        for i, _ in enumerate(new_ids):
            if new_ids[i] == id:
                lbl.append(new_labels[i])
                area.append(counts[candidates.index(new_labels[i])])
        splitting_labels.append(lbl)
        splitting_areas.append(area)

    merging_ids = []
    for lbl in merging_labels:
        id = []
        for i, _ in enumerate(new_labels):
            if new_labels[i] == lbl:
                id.append(new_ids[i])
        merging_ids.append(id)

    # logging.info(f"splitting_ids {splitting_ids}")
    # logging.info(f"splitting_labels {splitting_labels}")
    # logging.info(f"splitting_areas {splitting_areas}")

    # logging.info(f"merging_ids {merging_ids}")
    # logging.info(f"merging_labels {merging_labels}")

    cell_ids = []
    for cell in cells:
        cell_ids.append(cell.cell_id)

    # merging cells
    previously_timestepped_ids = []
    for j, ids in enumerate(merging_ids):
        for i, id in enumerate(ids):
            if id not in previously_timestepped_ids:
                previously_timestepped_ids.append(id)
                if i == 0:
                    cells[cell_ids.index(id)].label.append(merging_labels[j])
                    if merging_labels[j] in labels:
                        labels.remove(merging_labels[j])
                    cells[cell_ids.index(id)].datelist.append(nowdate)
                    cells[cell_ids.index(id)].area_gp.append(
                        counts[active_ids.index(id)]
                    )
                    cells[cell_ids.index(id)].score.append(scores[new_ids.index(id)])
                    cells[cell_ids.index(id)].overlap.append(
                        overlap[active_ids.index(id)]
                    )
                    # logging.info(
                # "cell %d was assigned label %d with %d count and %.3f score",
                #                         cells[cell_ids.index(id)].cell_id,
                #                         cells[cell_ids.index(id)].label[-1],
                #                         cells[cell_ids.index(id)].area_gp[-1],
                #                         cells[cell_ids.index(id)].score[-1],
                #                     )
                else:
                    cells[cell_ids.index(id)].merged_to = ids[0]
                    # logging.info(f"cell {id} was merged to id {ids[0]}")
                    cells[cell_ids.index(id)].terminate("merged")

    # splitting cells
    for j, id in enumerate(splitting_ids):
        areas = splitting_areas[j]
        label_candidates = [
            x for _, x in sorted(zip(areas, splitting_labels[j]), reverse=True)
        ]
        areas = sorted(areas, reverse=True)

        for i, label in enumerate(label_candidates):
            if i == 0:
                if id not in previously_timestepped_ids:
                    cells[cell_ids.index(id)].label.append(label)
                    if label in labels:
                        labels.remove(label)
                    cells[cell_ids.index(id)].datelist.append(nowdate)
                    cells[cell_ids.index(id)].area_gp.append(
                        counts[candidates.index(label)]
                    )
                    cells[cell_ids.index(id)].score.append(scores[new_ids.index(id)])
                    cells[cell_ids.index(id)].overlap.append(
                        overlap[candidates.index(label)]
                    )
                    # logging.info(
                    #     "cell %d was assigned label %d with %d count and %.3f score",
                    #     cells[cell_ids.index(id)].cell_id,
                    #     cells[cell_ids.index(id)].label[-1],
                    #     cells[cell_ids.index(id)].area_gp[-1],
                    #     cells[cell_ids.index(id)].score[-1],
                    # )
            else:
                cells[cell_ids.index(id)].child.append(cell_id)
                child_cells.append(Cell(cell_id, label, nowdate, parent=id))
                child_cells[-1].area_gp.append(counts[candidates.index(label)])
                # logging.info(
                #     "cell %d is child of %d with label %d and %d count",
                #     cell_id,
                #     id,
                #     label,
                #     cells[cell_ids.index(id)].area_gp[-1],
                # )
                cell_id += 1

    for cell in cells:
        if cell.cell_id in new_ids:
            if cell.cell_id in [item for sublist in merging_ids for item in sublist]:
                pass
            elif cell.cell_id in splitting_ids:
                pass
            else:
                new_idx = new_ids.index(cell.cell_id)
                idx = active_ids.index(cell.cell_id)

                new_label = new_labels[new_idx]

                cell.label.append(new_label)
                if new_label in labels:
                    labels.remove(new_label)
                cell.datelist.append(nowdate)
                cell.area_gp.append(counts[idx])
                # cells[mod_cell_id].area_gp.append(np.count_nonzero(labeled==new_label))
                cell.score.append(scores[new_idx])
                cell.overlap.append(overlap[idx])

                # logging.info(
                #     "cell %d was assigned label %d with %d count and %.3f score",
                #     cell.cell_id,
                #     cell.label[-1],
                #     cell.area_gp[-1],
                #     cell.score[-1],
                # )
        else:
            # logging.warning(f"cell with id {cell.cell_id} was not found in new step")
            cell.terminate("not found in new step (could be error)")

    # append child cells
    _ = [cells.append(child_cell) for child_cell in child_cells]

    return cells, labels, cell_id


def find_correspondences(
    active_ids,
    candidates,
    counts,
    overlap,
    last_active_area,
    last_velocity,
    last_center,
    candidate_center,
    alpha,
    velocity_score_weight,
    cluster_size_limit,
):
    """
    find all possible correspondences

    in
    active_ids: cell ids currently active, list
    candidates: future cell candidates, list of lists
    counts: number of gridpoints in cell candidates, list of lists
    overlap: overlap between active_ids and candidates, list of lists
    last_active_area: last active areas in gridpoints of cells: list
    last_velocity: last cell velocity, list
    last_center: last cell center, list
    candidate_center: center of candidate cells, list
    alpha: weight of overlap in score, float
    velocity_score_weight: weight of velocity in score, float
    cluster_size_limit: maximum number of cells in a cluster, before more crude solution is applied to solving cluster, int

    out
    new_ids_combined: new cell ids, list
    new_labels_combined: new cell labels, list
    scores_combined: scores of correspondences, list
    """
    active_ids_left = active_ids
    candidates_left = candidates

    new_ids_combined = []
    new_labels_combined = []
    scores_combined = []

    while len(active_ids_left) > 0:
        # address isolated clusters consecutively

        cluster_ids, cluster_labels = find_cluster_members(
            active_ids_left, candidates_left
        )
        # logging.info(f"cluster with ids {cluster_ids}, and labels {cluster_labels}")

        # find all possible combinations of correspondences between involved parties
        # calculate ratings for all possibilities

        new_ids, new_labels, score = correspond_cluster(
            cluster_ids,
            active_ids,
            candidates,
            counts,
            overlap,
            last_active_area,
            last_velocity,
            last_center,
            candidate_center,
            alpha,
            velocity_score_weight,
            cluster_size_limit,
        )

        new_ids_combined.extend(new_ids)
        new_labels_combined.extend(new_labels)
        scores_combined.extend(score)

        active_ids_left = [x for x in active_ids_left if x not in cluster_ids]
        candidates_left = [x for x in candidates_left if x not in cluster_labels]

    # logging.info(f"new_ids {new_ids_combined}")
    # logging.info(f"new_labels {new_labels_combined}")
    # logging.info(f"scores {scores_combined}")

    return new_ids_combined, new_labels_combined, scores_combined


def correspond_cluster(
    cluster_ids,
    active_ids,
    candidates_all,
    counts_all,
    overlap_all,
    last_active_area_all,
    last_velocity_all,
    last_center_all,
    candidate_center_all,
    alpha,
    velocity_score_weight,
    cluster_size_limit,
):
    """
    finds and scores correspondences of cluster

    in
    cluster_ids: cell ids in cluster, list
    active_ids: active cell ids, list
    candidates_all: labeled area cell candidates, list
    counts_all: gridpoint areas of new labeled areas, list
    overlap_all: overlap between tracked cells and candidates, list
    last_active_area_all: gridpoint areas of last timestep, list
    last_velocity_all: last cell velocity, list
    last_center_all: last cell center, list
    candidate_center_all: center of candidate cells, list
    alpha: weight of overlap in score, float
    velocity_score_weight: weight of velocity in score, float
    cluster_size_limit: maximum number of cells in cluster, int

    out
    new_ids: these ids can be assigned with new_labels, list
    new_labels: new labels to assign to new_ids, list
    scores_filtered: scores of the correspondences, list
    """
    ids = []
    candidates = []
    counts = []
    overlap = []
    last_active_area = []
    last_velocity = []
    last_center = []
    candidate_center = []

    for i, id in enumerate(active_ids):
        if id in cluster_ids:
            ids.append(id)
            candidates.append(candidates_all[i])
            counts.append(counts_all[i])
            overlap.append(overlap_all[i])
            last_active_area.append(last_active_area_all[i])
            last_velocity.append(last_velocity_all[i])
            last_center.append(last_center_all[i])
            candidate_center.append(candidate_center_all[i])

    # if candidaates are too long, shorten by means of overlap, so only the n candidates with highest overlap are considered
    if len(candidates) > cluster_size_limit:
        (
            ids,
            candidates,
            counts,
            overlap,
            last_active_area,
            last_velocity,
            last_center,
            candidate_center,
        ) = reduce_cluster_size(
            ids,
            candidates,
            counts,
            overlap,
            last_active_area,
            last_velocity,
            last_center,
            candidate_center,
            cluster_size_limit,
        )

    permutations = permutate_cluster(ids, candidates)
    scores = []
    mean_scores = []
    for perm in permutations:
        perm = list(perm)
        score = []
        for id in set(ids):
            mask = np.logical_and(perm, [id == i for i in ids])

            mask = list(itertools.compress(range(len(ids)), mask))
            if mask:
                score.append(
                    calculate_score(
                        [last_active_area[i] for i in mask],
                        [overlap[i] for i in mask],
                        [counts[i] for i in mask],
                        [last_velocity[i] for i in mask],
                        [last_center[i] for i in mask],
                        [candidate_center[i] for i in mask],
                        alpha,
                        velocity_score_weight,
                    )
                )
            else:
                score.append(0)

        scores.append(score)
        mean_scores.append(np.mean(score))

    # sort by best mean cluster score
    permutations = [x for _, x in sorted(zip(mean_scores, permutations), reverse=True)]
    scores = [x for _, x in sorted(zip(mean_scores, scores), reverse=True)]
    mean_scores = sorted(mean_scores, reverse=True)

    new_labels = []
    new_ids = []
    scores_filtered = []

    # chose best permutation
    for i, perm in enumerate(permutations[0]):
        if perm:
            new_ids.append(ids[i])
            new_labels.append(candidates[i])
            scores_filtered.append(scores[i][0])
        # else:
        #     new_labels.append(np.nan)

    return new_ids, new_labels, scores_filtered


def reduce_cluster_size(
    ids,
    candidates,
    counts,
    overlap,
    last_active_area,
    last_velocity,
    last_center,
    candidate_center,
    cluster_size_limit,
):
    """
    if cluster is too big, shorten by means of overlap, so only the n candidates with highest overlap are considered

    in
    ids: tracked cell ids in cluster, list
    candidates: cell candidates, list
    counts: number of gridpoints in cell candidates, list
    overlap: overlap between tracked cells and candidates, list
    last_active_area: gridpoint areas of last timestep, list
    last_velocity: last cell velocity, list
    last_center: last cell center, list
    candidate_center: center of candidate cells, list
    cluster_size_limit: maximum number of cells in cluster, int

    out
    ids: tracked cell ids in cluster, list
    candidates: cell candidates, list
    counts: number of gridpoints in cell candidates, list
    overlap: overlap between tracked cells and candidates, list
    last_active_area: gridpoint areas of last timestep, list
    last_velocity: last cell velocity, list
    last_center: last cell center, list
    candidate_center: center of candidate cells, list

    """
    orig_size = len(ids)
    orig_ids = set(ids)
    orig_candidates = set(candidates)

    while len(ids) > cluster_size_limit:
        # bool mask of dublicate candidates and ids
        mask_doubles_ids = np.isin(
            ids, np.unique(ids)[np.unique(ids, return_counts=True)[1] > 1]
        )
        mask_doubles_candidates = np.isin(
            candidates,
            np.unique(candidates)[np.unique(candidates, return_counts=True)[1] > 1],
        )
        mask_doubles = np.logical_and(mask_doubles_ids, mask_doubles_candidates)

        # remove the correspondance with the lowest overlap, only if it is not the only instance of that id or candidate
        if sum(mask_doubles) > 0:
            masked_overlap = np.array(overlap).astype(float)
            masked_overlap[~mask_doubles] = np.nan
            min_index = np.nanargmin(masked_overlap)
            ids.pop(min_index)
            candidates.pop(min_index)
            counts.pop(min_index)
            overlap.pop(min_index)
            last_active_area.pop(min_index)
            last_velocity.pop(min_index)
            last_center.pop(min_index)
            candidate_center.pop(min_index)
        else:
            break

    if len(ids) > cluster_size_limit:
        print(
            "warning: cluster with original size "
            + str(orig_size)
            + " could only be reduced to "
            + str(len(ids))
            + ", cluster_size_limit is "
            + str(cluster_size_limit)
            + ", chose different thresholds to reduce cluster sizes further"
        )

    new_ids = set(ids)
    new_candidates = set(candidates)

    # check if orig_ids and orig_candidates are still in new_ids and new_candidates
    if not orig_ids.issubset(new_ids):
        print("warning: original ids not in new ids")
        print(orig_ids)
        print(new_ids)
    if not orig_candidates.issubset(new_candidates):
        print("warning: original candidates not in new candidates")
        print(orig_candidates)
        print(new_candidates)
        exit()

    return (
        ids,
        candidates,
        counts,
        overlap,
        last_active_area,
        last_velocity,
        last_center,
        candidate_center,
    )


def permutate_cluster(ids, candidates):
    """
    permutates cluster in all possible ways

    in
    ids: tracked cell ids in cluster, list
    candidates: cell candidates, list

    out
    permutations: all possible permutations of correspondences, list
    """

    l = [False, True]
    permutations = list(itertools.product(l, repeat=len(ids)))

    # permutations = list(itertools.compress(range(len(ids)), permutations))

    permutations.pop(0)

    # should never be reached, since it is handeled by cluster_size_limit & reduce_cluster_size
    if len(ids) > 16:
        # logging.warning(
        #     f"large number of permutations: {len(permutations)}, from {len(ids)} ids"
        # )
        pass
    if len(ids) > 20:
        # todo: reaching this would still break the tracking, especially in parallel processing
        sys.exit(f"too many permutations: {len(permutations)}")

    return permutations


def find_cluster_members(active_ids, candidates):
    """
    finds all members of cluster containing first active_ids entry

    in
    active_ids: active tracked cell ids, list
    candidates: cell candidates, list

    out
    cluster_ids: cell ids in cluster, list
    cluster_labels: labels of candidates in cluster, list
    """
    cluster_ids = [active_ids[0]]
    cluster_labels = []
    old_length = -1
    new_length = 0
    while old_length < new_length:
        cluster_labels.extend(
            [candidates[i] for i, e in enumerate(active_ids) if e in cluster_ids]
        )
        cluster_labels = list(set(cluster_labels))
        cluster_ids.extend(
            [active_ids[i] for i, e in enumerate(candidates) if e in cluster_labels]
        )
        cluster_ids = list(set(cluster_ids))
        old_length = new_length
        new_length = len(cluster_ids) + len(cluster_labels)

    return cluster_ids, cluster_labels


def find_overlaps(
    cells,
    labeled,
    last_labeled,
    field,
    prev_field,
    flow_field,
    advection_method,
    dynamic_tracking,
    v_limit,
    min_area,
):
    """
    finds overlaping new labels from previous cells

    in
    cells: list of cell objects, list
    labeled: labeled cell areas, array
    last_labeled: last labeled cell areas, array
    field: 2d meteorological field, array
    prev_field: previous 2d meteorological field, array
    flow_field: estimate of flow field, array
    advection_method: method used to advect cells, string
    dynamic_tracking: number of timesteps used for advecting search mask, int
    v_limit: limit of distance (in gridpoints) per timestept to advect search mask, float
    min_area: minimum area (in gridpoints) for a cell to be considered, int

    out
    active_ids: cell ids currently active, list
    candidates: future cell candidates, list
    counts: number of gridpoints in cell candidates, list
    overlap: overlap between tracked cells and candidates, list
    last_active_area: gridpoint areas of last timestep, list
    last_velocity: last cell velocity, list
    last_center: last cell center, list
    """
    active_ids = []
    candidates = []
    counts = []
    overlap = []
    last_active_area = []
    last_velocity = []
    last_center = []

    # find areas overlapping with last timestep
    for cell in cells:  # pre-existing, alive cells in list
        last_active = last_labeled == cell.label[-1]

        new_delta_x, new_delta_y = determine_cell_movement(
            cell.delta_x,
            cell.delta_y,
            dynamic_tracking,
            v_limit,
        )

        if advection_method == "movement_vector":
            cell.search_vector.append([new_delta_x, new_delta_y])
            last_active = advect_array(
                flow_field, last_active, new_delta_x, new_delta_y
            )
        elif advection_method == "none":
            last_active = last_active

        laa = np.count_nonzero(last_active)

        cell.search_field.append(last_active)

        new_cell_labels = get_new_labels(labeled[last_active], min_area)

        ovl = []
        area_gp = []
        ids = []
        laas = []
        lvel = []
        lcent = []

        if new_cell_labels:
            for label in new_cell_labels:
                ovl.append(
                    np.count_nonzero(np.logical_and(labeled == label, last_active))
                )
                area_gp.append(np.sum(labeled == label))
                ids.append(cell.cell_id)
                laas.append(laa)
                lvel.append([new_delta_x, new_delta_y])
                lcent.append([cell.mass_center_x[-1], cell.mass_center_y[-1]])

            active_ids.extend(ids)
            candidates.extend(new_cell_labels)
            counts.extend(area_gp)
            overlap.extend(ovl)
            last_active_area.extend(laas)
            last_velocity.extend(lvel)  # todo
            last_center.extend(lcent)

        else:
            cell.terminate("faded")

    # logging.info(f"active_ids {active_ids}")
    # logging.info(f"candidates {candidates}")
    # logging.info(f"counts {counts}")
    # logging.info(f"overlap {overlap}")

    return (
        active_ids,
        candidates,
        counts,
        overlap,
        last_active_area,
        last_velocity,
        last_center,
    )


def calculate_score(
    last_active_area,
    overlap,
    area_gp,
    last_velocity,
    last_center,
    candidate_center,
    alpha,
    velocity_score_weight,
):
    """
    calculate correspondence score between old feature and all new features

    in
    last_active_area: gridpoint areas of last timestep, list
    overlap: overlap between tracked cells and candidates, list
    area_gp: gridpoint areas of new labeled areas, list
    last_velocity: last cell velocity, list
    last_center: last cell center, list
    candidate_center: center of candidate cells, list
    alpha: weight of overlap in score, float
    velocity_score_weight: weight of velocity in score, float

    out
    score: score of correspondence, float
    """
    # adapted from diss ruedishueli
    alpha = alpha
    a_p = last_active_area[0]
    a_c = np.sum(area_gp)
    a_o = np.sum(overlap)

    r_o = a_o / np.minimum(a_p, a_c)  # overlap ratio
    r_s = np.minimum(a_p, a_c) / np.maximum(a_p, a_c)  # surface ratio

    p_t = alpha * r_o + (1 - alpha) * r_s
    area_score = p_t

    # score based on velocity similarity from last to current timestep

    beta = velocity_score_weight

    if (None in [item for sublist in last_velocity for item in sublist]) | (
        None in [item for sublist in last_center for item in sublist]
    ):
        score = area_score
    else:
        velocity_score = []
        for i, _ in enumerate(last_velocity):
            candidate_velocity = np.subtract(candidate_center[i], last_center[i])
            estimate_magnitude = np.sqrt(
                last_velocity[i][0] ** 2 + last_velocity[i][1] ** 2
            )
            estimate_error = np.subtract(candidate_velocity, last_velocity[i])
            estimate_error_magnitude = np.sqrt(
                estimate_error[0] ** 2 + estimate_error[1] ** 2
            )
            if estimate_magnitude == 0:
                velocity_score.append(0)
            else:
                velocity_score.append(
                    np.maximum(-estimate_error_magnitude / estimate_magnitude + 1, 0)
                )
        score = (1 - beta) * area_score + beta * np.mean(velocity_score)
    return score


def remap_ids(cells):
    """
    remaps cell ids after filtering so consecutive int's are used

    in
    cells: list of cell objects, list

    out
    cells: list of cell objects, list
    """
    cell_id = 0
    new_ids = []
    old_ids = []

    for cell in cells:
        new_ids.append(cell_id)
        old_ids.append(cell.cell_id)
        cell.cell_id = cell_id
        cell_id += 1

    for cell in cells:
        if cell.parent in old_ids:  # if cell has parent which is still in filtered list
            cell.parent = new_ids[old_ids.index(cell.parent)]
        else:
            cell.parent = None

        if cell.merged_to in old_ids:
            cell.merged_to = new_ids[old_ids.index(cell.merged_to)]
        else:
            cell.merged_to = None

        new_child = []
        for child in cell.child:
            if child in old_ids:
                new_child.append(new_ids[old_ids.index(child)])
        cell.child = new_child

    return cells


def find_events(cells):
    """
    finds splitting events and returns summary thereof
    in
    cells: list of cell objects, list

    out

    """

    for cell in cells:
        if cell.parent:  # cell originates from splitting as it has a parent
            print(
                "cell",
                cell.cell_id,
                "is child of",
                cell.parent,
                "split occurred at",
                cell.datelist[0],
            )

        if cell.merged_to is not None:
            print(
                "cell",
                cell.cell_id,
                "merged to",
                cell.merged_to,
                "merger occurred at",
                cell.datelist[-1],
            )


def get_new_labels(arr, min_area):
    """
    returns new labels that overlay with old labels sorted with ascending count

    in
    arr: labeled array, array
    min_area: minimum area (in gridpoints) for a cell to be considered, int

    out
    new_labels_filtered: new labels that overlay with old labels sorted with ascending count, list
    """
    new_labels = np.unique(arr).tolist()
    new_labels = [i for i in new_labels if i != 0]  # remove zero
    count_filtered = []
    new_labels_filtered = []

    for label in new_labels:
        count = np.count_nonzero(label == arr)
        if count >= min_area:
            count_filtered.append(count)
            new_labels_filtered.append(label)

    # sort lists
    _ = [
        label
        for _, label in sorted(zip(count_filtered, new_labels_filtered), reverse=True)
    ]
    # count_filtered = sorted(count_filtered, reverse=True)

    return new_labels_filtered


def interpolate_footprints(
    cell_values,
    last_cell_values,
    cell_coordinates,
    last_cell_coordinates,
    delta_x,
    delta_y,
    supersampling,
):
    """
    interpolates two cell timestep footprints linearly

    in
    cell_values: values of cell in current timestep, array
    last_cell_values: values of cell in last timestep, array
    cell_coordinates: coordinates of cell in current timestep, array
    last_cell_coordinates: coordinates of cell in last timestep, array
    delta_x: movement vector of cell mass center in x direction, float
    delta_y: movement vector of cell mass center in y direction, float
    supersampling: number of steps to interpolate, int

    out
    array: interpolated array, array
    min_coords: minimum coordinates of interpolated array, array
    max_coords: maximum coordinates of interpolated array, array
    """
    padding = int(np.ceil(np.max((abs(delta_x), abs(delta_y)))) + 2)
    min_coords = (
        np.amin(np.concatenate((cell_coordinates, last_cell_coordinates)), axis=0)
        - padding
    )
    max_coords = (
        np.amax(np.concatenate((cell_coordinates, last_cell_coordinates)), axis=0)
        + padding
        + 1
    )
    new_coords = cell_coordinates - min_coords
    new_last_coords = last_cell_coordinates - min_coords
    array = np.zeros((max_coords[0] - min_coords[0], max_coords[1] - min_coords[1]))
    for i in range(supersampling + 1):
        blend = np.zeros_like(array)
        weight = (supersampling - i) / supersampling
        last_weight = i / supersampling
        x_shift = int(np.round(delta_x / supersampling * i))
        y_shift = int(np.round(delta_y / supersampling * i))
        last_x_shift = x_shift - int(np.round(delta_x))
        last_y_shift = y_shift - int(np.round(delta_y))
        blend[new_coords[:, 0] - x_shift, new_coords[:, 1] - y_shift] = (
            cell_values * weight
        )
        blend[
            new_last_coords[:, 0] - last_x_shift, new_last_coords[:, 1] - last_y_shift
        ] += (last_cell_values * last_weight)
        array = np.max(np.dstack((array, blend)), 2)
    return array, min_coords, max_coords


def make_aura_kernel(radius: int):
    """
    returns 2d binary structure with radius

    in
    radius: radius of structure, int

    out
    kernel: 2d binary structure with radius, array
    """
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            if np.sqrt((i - radius) ** 2 + (j - radius) ** 2) <= radius:
                kernel[i, j] = 1
    return kernel


class Cell:
    """
    cell class with atributes for each cell
    """

    def __init__(self, cell_id: int, label: int, nowdate, parent=None):
        """
        inizialize cell object

        in
        cell_id: cell id, int
        label: label of cell, int
        nowdate: current date, datetime
        parent: parent cell id, int
        """
        self.cell_id = cell_id
        self.label = []
        self.label.append(label)
        self.datelist = []
        self.datelist.append(nowdate)
        self.alive = True
        self.parent = parent
        self.child = []
        self.merged_to = None

        self.mass_center_x = []  # list of x positions of mass center
        self.mass_center_y = []  # list of y positions of mass center

        self.delta_x = []  # movement vector of cell mass center
        self.delta_y = []  # movement vector of cell mass center

        self.lon = []
        self.lat = []

        self.area_gp = [] # area in gridpoints
        self.width_gp = [] # width of cell footprint in gridpoints
        self.length_gp = [] # length of cell footprint in gridpoints
        self.max_val = []
        self.max_x = []
        self.max_y = []

        self.lifespan = None
        self.score = [0]  # first assignment doesn't have a score
        self.overlap = [0]  # first assignment doesn't have a overlap

        self.died_of = None

        self.field = [] # list of coordinates that are inside the cell mask

        self.swath = None

        # field of area where cell is searched for (extrapolated from last step, first is None)
        self.search_field = [None]
        self.search_vector = [[0, 0]]

    def __str__(self):
        """
        default string representation of Cell object
        """
        return str(self.cell_id)

    def cal_spatial(self, field_static, coordinates, values):
        """
        calculate spatial stats (to be applied per timestep)
        used on calculations that rely on the field and labeled arrays only available
        at runtime

        in
        field_static: static field, dict
        coordinates: coordinates that lie within cell, array
        values: values of cell, array
        """
        self.field.append(coordinates)

        # max_pos = np.unravel_index(masked.argmax(),masked.shape)
        area_center = np.mean(coordinates, axis=0)
        max_pos = coordinates[np.argmax(values)]

        mass_center = np.mean((area_center, max_pos), 0)

        # max_pos = ndimage.measurements.maximum_position(masked)
        # mass_center = max_pos
        # mass_center = ndimage.measurements.center_of_mass(field,labeled,self.label[-1])

        self.mass_center_x.append(mass_center[0])
        self.mass_center_y.append(mass_center[1])

        if len(self.mass_center_x) < 2:
            self.delta_x.append(0)
            self.delta_y.append(0)

            self.width_gp.append(0)
            self.length_gp.append(0)
        else:
            self.delta_x.append(self.mass_center_x[-1] - self.mass_center_x[-2])
            self.delta_y.append(self.mass_center_y[-1] - self.mass_center_y[-2])

            # get width and length relative to movement vector
            # rotate the cell mask 
            angle = np.arctan2(self.delta_y[-1], self.delta_x[-1]) * 180 / np.pi

            mask_size = np.max(coordinates, axis=0)-np.min(coordinates, axis=0)+1

            mask = np.zeros(mask_size)
            mask[coordinates[:,0]-np.min(coordinates, axis=0)[0], coordinates[:,1]-np.min(coordinates, axis=0)[1]] = 1

            cell_mask_r = rotate(mask, angle)

            # threshold the rotated cell mask
            cell_mask_r[cell_mask_r<0.5] = 0
            cell_mask_r[cell_mask_r>0.5] = 1

            width_gp = np.where(cell_mask_r)[1].max()-np.where(cell_mask_r)[1].min()
            length_gp = np.where(cell_mask_r)[0].max()-np.where(cell_mask_r)[0].min()

            self.width_gp.append(width_gp)
            self.length_gp.append(length_gp)

        # self.area_gp.append(np.count_nonzero(masked)) # area in gridpoints
        self.max_val.append(np.max(values))  # maximum value of field

        self.max_x.append(max_pos[0])
        self.max_y.append(max_pos[1])

        self.lon.append(
            field_static["lon"][
                int(np.round(self.mass_center_x[-1])),
                int(np.round(self.mass_center_y[-1])),
            ]
        )
        self.lat.append(
            field_static["lat"][
                int(np.round(self.mass_center_x[-1])),
                int(np.round(self.mass_center_y[-1])),
            ]
        )

    def is_in_box(self, lat_lim, lon_lim):
        """
        check if any part of cell is in lat lon box

        in
        lat_lim: latitude limits, list
        lon_lim: longitude limits, list
        """

        if np.any(
            np.logical_and(
                np.array(self.lon) > lon_lim[0], np.array(self.lon) < lon_lim[1]
            )
        ):
            if np.any(
                np.logical_and(
                    np.array(self.lat) > lat_lim[0], np.array(self.lat) < lat_lim[1]
                )
            ):
                return True
        return False

    def post_processing(self):
        """
        post process cells
        used on calculations that are performed after tracking
        """

        self.lifespan = self.datelist[-1] - self.datelist[0]

    def insert_split_timestep(self, parent):
        """
        insert timestep from parent cell

        in
        parent: parent cell, Cell
        """

        if self.datelist[-1] not in parent.datelist:
            # logging.warning(
            #     "parent datelist does not contain self datelist[-1], could not append split member"
            # )
            return

        split_idx = parent.datelist.index(self.datelist[0]) - 1
        self.label.insert(0, parent.label[split_idx])

        self.datelist.insert(0, parent.datelist[split_idx])

        self.mass_center_x.insert(0, parent.mass_center_x[split_idx])
        self.mass_center_y.insert(0, parent.mass_center_y[split_idx])

        self.delta_x.insert(0, parent.delta_x[split_idx])
        self.delta_y.insert(0, parent.delta_y[split_idx])

        self.lon.insert(0, parent.lon[split_idx])
        self.lat.insert(0, parent.lat[split_idx])

        self.area_gp.insert(0, parent.area_gp[split_idx])
        self.max_val.insert(0, parent.max_val[split_idx])
        self.max_x.insert(0, parent.max_x[split_idx])
        self.max_y.insert(0, parent.max_y[split_idx])

        self.score.insert(0, parent.score[split_idx])
        self.overlap.insert(0, parent.overlap[split_idx])

        self.field.insert(0, parent.field[split_idx])

        self.search_field.insert(0, None)
        self.search_vector.insert(0, [0, 0])

    def insert_merged_timestep(self, parent):
        """
        insert timestep from merged_to cell

        in
        parent: parent cell, Cell
        """
        # logging.info(
        # "inserting merged timestep from "
        #     + str(parent.cell_id)
        #     + " to "
        #     + str(self.cell_id)
        # )

        if self.datelist[-1] not in parent.datelist:
            # logging.warning(
            #     "parent datelist does not contain self datelist[-1], could not append merge member"
            # )
            return

        merge_idx = parent.datelist.index(self.datelist[-1])  # + 1 # todo
        self.label.append(parent.label[merge_idx])  # todo

        self.datelist.append(parent.datelist[merge_idx])

        self.mass_center_x.append(parent.mass_center_x[merge_idx])
        self.mass_center_y.append(parent.mass_center_y[merge_idx])

        self.delta_x.append(parent.delta_x[merge_idx])
        self.delta_y.append(parent.delta_y[merge_idx])

        self.lon.append(parent.lon[merge_idx])
        self.lat.append(parent.lat[merge_idx])

        self.area_gp.append(parent.area_gp[merge_idx])
        self.max_val.append(parent.max_val[merge_idx])
        self.max_x.append(parent.max_x[merge_idx])
        self.max_y.append(parent.max_y[merge_idx])

        self.score.append(parent.score[merge_idx])
        self.overlap.append(parent.overlap[merge_idx])

        self.field.append(parent.field[merge_idx])

        self.search_field.append(None)
        self.search_vector.append([0, 0])

    def terminate(self, reason=None):
        """
        terminates cell

        in
        reason: reason for termination, string
        """
        if self.alive:
            self.died_of = reason
            self.alive = False
            # logging.info("cell %d was terminated, reason is: %s", self.cell_id, reason)
        else:
            pass
            # logging.warning("cell %d is already dead!", self.cell_id)

    def print_summary(self):
        """
        diagnostics function
        """
        print(
            "cell no.",
            str(self.cell_id).zfill(3),
            " started at",
            np.round(self.lat[0], 2),
            "N",
            np.round(self.lon[0], 2),
            "E",
            self.datelist[0],
            "lasted",
            str(self.lifespan)[0:-3],
            "h",
        )
        print(
            "              max_area:",
            max(self.area_gp),
            "max_val:",
            round(np.max(self.max_val), 2),
        )
        print(
            "              parent:",
            self.parent,
            "childs:",
            self.child,
            "merged to:",
            self.merged_to,
        )

    def add_aura(self, kernel_size, field_static):
        """
        add aura to cell
        used for plotting and analysis

        in
        kernel_size: size of aura, int
        field_static: static field, dict
        """
        if kernel_size == 0:
            return

        kernel = make_aura_kernel(kernel_size)

        for i, field in enumerate(self.field):
            array = np.zeros_like(field_static["lat"])
            array[field[:, 0], field[:, 1]] = 1
            array = ndimage.binary_dilation(array, structure=kernel)
            self.field[i] = np.argwhere(array == 1)

    def add_genesis(self, n_timesteps, field_static):
        """
        add genesis timesteps to cell
        interpolated backwards in time to where the cell is initially detected
        mainly used for plotting and analysis

        in
        n_timesteps: number of timesteps to add, int
        field_static: static field, dict
        """
        x_0 = self.mass_center_x[0].copy()
        y_0 = self.mass_center_y[0].copy()
        field_0 = self.field[0].copy()

        dx = np.mean(self.delta_x[1:4])
        dy = np.mean(self.delta_y[1:4])

        self.delta_x[0] = dx
        self.delta_y[0] = dy

        for i in range(1, n_timesteps):
            x_offset = int(np.round(dx * i))
            y_offset = int(np.round(dy * i))

            self.delta_x.insert(0, self.delta_x[0])
            self.delta_y.insert(0, self.delta_y[0])

            self.datelist.insert(
                0, self.datelist[0] - (self.datelist[1] - self.datelist[0])
            )

            self.mass_center_x.insert(0, x_0 - x_offset)
            self.mass_center_y.insert(0, y_0 - y_offset)

            new_field = np.zeros_like(field_0)
            new_field[:, 0] = field_0[:, 0] - x_offset
            new_field[:, 1] = field_0[:, 1] - y_offset
            self.field.insert(0, new_field)

            self.lon.insert(
                0,
                field_static["lon"][
                    int(np.round(self.mass_center_x[0])),
                    int(np.round(self.mass_center_y[0])),
                ],
            )
            self.lat.insert(
                0,
                field_static["lat"][
                    int(np.round(self.mass_center_x[0])),
                    int(np.round(self.mass_center_y[0])),
                ],
            )

            self.area_gp.insert(0, self.area_gp[0])

            self.label.insert(0, None)
            self.max_val.insert(0, None)
            self.max_x.insert(0, None)
            self.max_y.insert(0, None)
            self.score.insert(0, None)
            self.overlap.insert(0, None)
            self.search_field.insert(0, None)
            self.search_vector.insert(0, [0, 0])

    def append_associates(self, cells, field_static=None):
        """
        appends all associated cells to self cell

        in
        cells: list of cell objects, list
        field_static: static field, dict
        """
        cell_ids = [cell.cell_id for cell in cells]

        if self.parent is not None:
            if self.parent in cell_ids:
                self.append_cell(cells[self.parent], field_static)

        if self.child is not None:
            for child in self.child:
                if child in cell_ids:
                    self.append_cell(cells[child], field_static)

        if self.merged_to is not None:
            if self.merged_to in cell_ids:
                self.append_cell(cells[self.merged_to], field_static)

    def append_cell(self, cell_to_append, field_static=None):
        """
        appends cell_to_append to self cell
        used to append short lived cells to parent
        todo: delta_x/y not recalculated

        in
        cell_to_append: cell to append, Cell
        field_static: static field, dict
        """

        for i, nowdate in enumerate(cell_to_append.datelist):
            # both cells exist simultaneously

            if nowdate in self.datelist:
                index = self.datelist.index(nowdate)

                if self.field is not None:
                    self.field[index] = np.append(
                        self.field[index], cell_to_append.field[i], axis=0
                    )

                    self.field[index] = np.unique(self.field[index], axis=0)

                    mass_center = np.mean(self.field[index], axis=0)

                    self.mass_center_x[index] = mass_center[0]
                    self.mass_center_y[index] = mass_center[1]

                    self.area_gp[index] = np.shape(self.field[index])[
                        0
                    ]  # area in gridpoints

                else:
                    # weighted mean approximation
                    self.mass_center_x[index] = cell_to_append.mass_center_x[
                        i
                    ] * cell_to_append.area_gp[i] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    ) + self.mass_center_x[
                        index
                    ] * self.area_gp[
                        index
                    ] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    )
                    self.mass_center_y[index] = cell_to_append.mass_center_y[
                        i
                    ] * cell_to_append.area_gp[i] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    ) + self.mass_center_y[
                        index
                    ] * self.area_gp[
                        index
                    ] / (
                        cell_to_append.area_gp[i] + self.area_gp[index]
                    )

                    # first or last timestep of cell_to_append has been added by add_split_timestep or add_merged_timestep so it doesnt need to be added.
                    if i != 0 and i != len(cell_to_append.datelist) - 1:
                        self.area_gp[index] += cell_to_append.area_gp[i]

                if self.max_val[index] < cell_to_append.max_val[i]:
                    self.max_val[index] = cell_to_append.max_val[i]
                    self.max_x[index] = cell_to_append.max_x[i]
                    self.max_y[index] = cell_to_append.max_y[i]

                if field_static is not None:
                    self.lon[index] = field_static["lon"][
                        int(np.round(self.mass_center_x[index])),
                        int(np.round(self.mass_center_y[index])),
                    ]
                    self.lat[index] = field_static["lat"][
                        int(np.round(self.mass_center_x[index])),
                        int(np.round(self.mass_center_y[index])),
                    ]

            elif nowdate > self.datelist[-1]:
                index = -1
                if self.field is not None:
                    self.field.append(cell_to_append.field[i])

                self.datelist.append(nowdate)

                self.mass_center_x.append(cell_to_append.mass_center_x[i])
                self.mass_center_y.append(cell_to_append.mass_center_y[i])

                self.delta_x.append(cell_to_append.delta_x[i])
                self.delta_y.append(cell_to_append.delta_y[i])

                # area in gridpoints
                self.area_gp.append(cell_to_append.area_gp[i])
                # maximum value of field
                self.max_val.append(cell_to_append.max_val[i])

                self.max_x.append(cell_to_append.max_x[i])
                self.max_y.append(cell_to_append.max_y[i])

                self.lon.append(cell_to_append.lon[i])
                self.lat.append(cell_to_append.lat[i])

                self.label.append(-1)  # nan representation
                self.score.append(-1)  # nan representation
                self.overlap.append(-1)  # nan representation
                self.search_field.append(None)
                self.search_vector.append(None)

                self.lifespan = self.datelist[-1] - self.datelist[0]

            elif nowdate < self.datelist[0]:
                index = -1
                if self.field is not None:
                    self.field.insert(0, cell_to_append.field[i])
                self.datelist.insert(0, nowdate)

                self.mass_center_x.insert(0, cell_to_append.mass_center_x[i])
                self.mass_center_y.insert(0, cell_to_append.mass_center_y[i])

                self.delta_x.insert(0, cell_to_append.delta_x[i])
                self.delta_y.insert(0, cell_to_append.delta_y[i])

                # area in gridpoints
                self.area_gp.insert(0, cell_to_append.area_gp[i])
                self.max_val.insert(
                    0, cell_to_append.max_val[i]
                )  # maximum value of field

                self.max_x.insert(0, cell_to_append.max_x[i])
                self.max_y.insert(0, cell_to_append.max_y[i])

                self.lon.insert(0, cell_to_append.lon[i])
                self.lat.insert(0, cell_to_append.lat[i])

                self.label.insert(0, -1)  # nan representation
                self.score.insert(0, -1)  # nan representation
                self.overlap.insert(0, -1)  # nan representation
                self.search_field.insert(0, None)
                self.search_vector.insert(0, None)

                self.lifespan = self.datelist[-1] - self.datelist[0]

            else:
                print("other appending error")

    def copy(self):
        """
        returns a copy of the cell object
        """
        return copy.deepcopy(self)

    def get_human_str(self):
        """
        generates human readable string of cell object for # logging

        out
        outstr: human readable string of cell object, string
        """
        outstr = "-" * 100 + "\n"
        outstr += f"cell_id:  {str(self.cell_id).zfill(3)}\n"

        outstr += f"start:      {self.datelist[0]}\n"
        outstr += f"end:        {self.datelist[-1]}\n"
        outstr += f"lifespan:   {self.lifespan.astype('timedelta64[m]')}\n"
        outstr += f"parent:     {self.parent}\n"
        outstr += f"childs:     {self.child}\n"
        outstr += f"merged:     {self.merged_to}\n"
        outstr += f"died_of:    {self.died_of}\n"

        outstr += "\n"

        outstr += f"x:          {list(np.round(self.mass_center_x,2))}\n"
        outstr += f"y:          {list(np.round(self.mass_center_y,2))}\n"
        outstr += f"delta_x:    {list(np.round(self.delta_x,2))}\n"
        outstr += f"delta_y:    {list(np.round(self.delta_y,2))}\n"
        outstr += f"lon:        {list(np.round(self.lon,3))}\n"
        outstr += f"lat:        {list(np.round(self.lat,3))}\n"
        outstr += f"area_gp:    {self.area_gp}\n"
        outstr += f"max_val:    {list(np.round(self.max_val,2))}\n"
        outstr += f"score:      {list(np.round(self.score,3))}\n"
        outstr += f"overlap:    {self.overlap}\n"

        outstr += "\n\n"

        return outstr

    def check_consistency(self):
        """
        checks some cell parameters to see if they're cconsistent
        """
        if len(self.datelist) != len(self.mass_center_x):
            # logging.warning("cell %d has inconsistencies", self.cell_id)
            # logging.info("len(self.datelist): %d", len(self.datelist))
            # logging.info("len(self.mass_center_x): %d", len(self.mass_center_x))
            # logging.info("len(self.area_gp): %d", len(self.area_gp))
            # logging.debug(f"self.datelist {self.datelist}")
            # logging.debug(f"self.mass_center_x {self.mass_center_x}")
            # logging.debug(f"self.mass_center_y {self.mass_center_y}")
            # logging.debug(f"self.delta_x {self.delta_x}")
            # # logging.debug(f"self.delta_y {self.delta_y}")
            pass

    def to_dict(self):
        """
        returns a dictionary containing all cell object information

        out
        cell_dict: dictionary containing all cell object information, dict
        """
        cell_dict = {
            "cell_id": self.cell_id,
            "parent": self.parent,
            "child": self.child,
            "merged_to": self.merged_to,
            "died_of": self.died_of,
            "lifespan": self.lifespan / np.timedelta64(1, "m"),
            "datelist": [str(t)[:16] for t in self.datelist],
            "lon": [round(float(x), 4) for x in self.lon],
            "lat": [round(float(x), 4) for x in self.lat],
            "mass_center_x": [round(float(x), 2) for x in self.mass_center_x],
            "mass_center_y": [round(float(x), 2) for x in self.mass_center_y],
            "max_x": [int(x) for x in self.max_x],
            "max_y": [int(x) for x in self.max_y],
            "delta_x": [round(float(x), 2) for x in self.delta_x],
            "delta_y": [round(float(x), 2) for x in self.delta_y],
            "area_gp": [int(x) for x in self.area_gp],
            "max_val": [round(float(x), 2) for x in self.max_val],
            "score": [round(float(x), 2) for x in self.score],
        }
        return cell_dict

    def from_dict(self, cell_dict):
        """
        returns a cell object from a dictionary containing all cell object information

        in
        cell_dict: dictionary containing all cell object information, dict
        """
        self.cell_id = cell_dict["cell_id"]
        self.parent = cell_dict["parent"]
        self.child = cell_dict["child"]
        self.merged_to = cell_dict["merged_to"]
        self.died_of = cell_dict["died_of"]
        self.lifespan = cell_dict["lifespan"]
        self.datelist = [np.datetime64(t) for t in cell_dict["datelist"]]
        self.lon = cell_dict["lon"]
        self.lat = cell_dict["lat"]
        self.mass_center_x = cell_dict["mass_center_x"]
        self.mass_center_y = cell_dict["mass_center_y"]
        self.max_x = cell_dict["max_x"]
        self.max_y = cell_dict["max_y"]
        self.delta_x = cell_dict["delta_x"]
        self.delta_y = cell_dict["delta_y"]
        self.area_gp = cell_dict["area_gp"]
        self.max_val = cell_dict["max_val"]
        self.score = cell_dict["score"]

        self.overlap = [0] * len(self.datelist)
        self.search_field = [None] * len(self.datelist)
        self.search_vector = [[0, 0]] * len(self.datelist)
        self.alive = False
        self.field = None
        self.label = []
        self.swath = []


def write_to_json(cellss, filename):
    """
    writes ascii file containing cell object information
    cellss can be either list of cells, or list of list of cells (ensemble)

    in
    cellss: list of cells or list of list of cells, list
    filename: filename, string

    out
    struct: dictionary containing all cell object information, dict
    """

    # figure out wether input is ensemble or not
    if len(cellss) == 0:
        struct = []
        data_structure = "no cells found"

    elif isinstance(cellss[0], Cell):  # single member
        struct = [cell.to_dict() for cell in cellss]
        data_structure = "cell_data contains list of cells, each cell contains a a dictionary of cell parameters which are lists with length of the lifetime of the cell"

    elif isinstance(cellss[0], list):  # ensemble
        struct = [[cell.to_dict() for cell in cells] for cells in cellss]
        data_structure = "cell_data contains list of ensemle members, each of which is a list of cells, each cell contains a a dictionary of cell parameters which are lists with length of the lifetime of the cell"

    else:
        print(
            "input is neither list of cells nor list of list of cells, each cell contains a a dictionary of cell parameters which ar lists with length of the lifetime of the cell"
        )
        return

    struct = {
        "info": "cell track data generated using track_cells_v2.py",
        "author": "Killian P. Brennan (killian.brennan@env.ethz.ch)",
        "data_structure": data_structure,
        "parameters": {
            "cell_id": "unique identifier for each cell",
            "parent": "cell_id of parent cell, if any",
            "child": "list of cell_ids of child cells, if any",
            "merged_to": "cell_id of cell that this cell merged to, if any",
            "died_of": "cell_id of cell that this cell died of, if any",
            "lifespan": "lifespan of cell in minutes",
            "datelist": "list of datetimes for each timestep in cell lifetime",
            "lon": "list of longitudes for each timestep in cell lifetime (mass center)",
            "lat": "list of latitudes for each timestep in cell lifetime (mass center)",
            "mass_center_x": "list of mass center x coordinates for each timestep in cell lifetime",
            "mass_center_y": "list of mass center y coordinates for each timestep in cell lifetime",
            "max_x": "list of x coordinates of max value for each timestep in cell lifetime",
            "max_y": "list of y coordinates of max value for each timestep in cell lifetime",
            "delta_x": "list of delta x for each timestep in cell lifetime",
            "delta_y": "list of delta y for each timestep in cell lifetime",
            "area_gp": "list of area in gridpoints for each timestep in cell lifetime",
            "max_val": "list of max value for each timestep in cell lifetime",
            "score": "list of tracking score for each timestep in cell lifetime, -1 is nan",
        },
        "cell_data": struct,
    }

    with open(filename, "w") as f:
        json.dump(struct, f)

    return struct


def write_masks_to_netcdf(
    cellss,
    datelist,
    field_static,
    filename,
):
    """
    writes netcdf file containing cell masks
    cellss can be either list of cells, or list of list of cells (ensemble)
    field_static must include lat and lon 2d fields

    in
    cellss: list of cells or list of list of cells, list
    datelist: list of datetimes, list
    field_static: static field, dict
    filename: filename, string

    out
    ds: xarray dataset containing cell masks, xarray dataset
    """

    # figure out wether input is ensemble or not
    if len(cellss) == 0:
        members = [0]
        cellss = [cellss]

    elif isinstance(cellss[0], Cell):
        members = [0]
        cellss = [cellss]

    elif isinstance(cellss[0], list):
        members = np.arange(len(cellss))

    else:
        print("input is neither list of cells nor list of list of cells")
        return

    mask_array = np.zeros(
        (
            len(members),
            len(datelist),
            field_static["lat"].shape[0],
            field_static["lon"].shape[1],
        )
    )
    mask_array[:] = np.nan

    for m in members:
        for cell in cellss[m]:
            for i, t in enumerate(cell.datelist):
                date_index = np.where(datelist == t)[0][0]
                mask_array[
                    m, date_index, cell.field[i][:, 0], cell.field[i][:, 1]
                ] = cell.cell_id

    if len(members) == 1:
        # if we are not dealing with a ensemble, remove member dimension
        mask_array = mask_array[0]
        coords = {
            "time": datelist,
            "y": np.arange(field_static["lat"].shape[0]),
            "x": np.arange(field_static["lon"].shape[1]),
        }
        data_structure = {
            "cell_mask": (["time", "y", "x"], mask_array),
            "lat": (["y", "x"], field_static["lat"]),
            "lon": (["y", "x"], field_static["lon"]),
        }
    else:
        coords = {
            "member": members,
            "time": datelist,
            "y": np.arange(field_static["lat"].shape[0]),
            "x": np.arange(field_static["lon"].shape[1]),
        }
        data_structure = {
            "cell_mask": (["member", "time", "y", "x"], mask_array),
            "lat": (["y", "x"], field_static["lat"]),
            "lon": (["y", "x"], field_static["lon"]),
        }

    # create netcdf file
    ds = xr.Dataset(
        data_structure,
        coords=coords,
    )

    # write to netcdf file
    ds.to_netcdf(filename)

    # compress netcdf file
    # os.system("nczip " + filename)

    return ds

def read_from_json(filename):
    """
    reads cell objects from json file
    this is only a post processing function, not used in tracking
    """

    with open(filename, "r") as f:
        struct = json.load(f)

    if (
        struct["data_structure"]
        == "cell_data contains list of cells, each cell contains a a dictionary of cell parameters which are lists with length of the lifetime of the cell"
    ):
        cellss = []
        for cell_dict in struct["cell_data"]:
            cell = Cell(None, None, None)
            cell.from_dict(cell_dict)
            cellss.append(cell)

    elif (
        struct["data_structure"]
        == "cell_data contains list of ensemle members, each of which is a list of cells, each cell contains a a dictionary of cell parameters which are lists with length of the lifetime of the cell"
    ):
        cellss = []
        for member in struct["cell_data"]:
            cells = []
            for cell_dict in member:
                cell = Cell(None, None, None)
                cell.from_dict(cell_dict)
                cells.append(cell)
            cellss.append(cells)

    elif struct["data_structure"] == "no cells found":
        return []
    
    else:
        print("data structure not recognized")
        return

    return cellss

def add_masks(cells, filename):
    '''
    adds masks to cell objects from netcdf file
    this is only a post processing function, not used in tracking
    todo: this needs to be implemented
    '''
    return


if __name__ == "__main__":
    pass
