"""
tracks output files from cosmo-1e hindcasts
tracks features on column max of W


input:
    inpath: path to input files
    outpath: path to output files

output:
    pickle of cells
    json of cell tracks
    netcdf of cell masks

example use:
python /home/kbrennan/cell_tracker/track_case_study.py /home/kbrennan/phd/data/case_20210628 /home/kbrennan/phd/data/case_20210628/tracks_new

"""

import os
import argparse

import xarray as xr
import numpy as np

import _pickle as cPickle


from cell_tracker import (
    Cell,
    track_cells,
    write_to_json,
    write_masks_to_netcdf,
)

def main(inpath, outpath, paralel=False):

    # make sure output directory exists
    os.makedirs(outpath, exist_ok=True)

    # create list of members
    members = [str(m).zfill(3) for m in range(11)]
    # members = ['004']

    # iterate over member (todo: paralelize)
    if paralel:
        import multiprocessing as mp
        pool = mp.Pool(len(members))
        pool.starmap(pipeline, [(member, inpath, outpath) for member in members])
    else:
        for member in members:
            pipeline(member, inpath, outpath)
    
def pipeline(member, inpath, outpath):
    # print("processing member: ", member)

    # load data
    memberpath = os.path.join(inpath, member)

    ds = xr.open_mfdataset(
        os.path.join(memberpath, "lfff*.nc"), combine="by_coords"
    )

    field_static = {}
    field_static["lat"] = ds["lat"].values
    field_static["lon"] = ds["lon"].values

    timesteps = ds["time"].values

    # fields = ds["W"].max(dim="level1").values # todo: leads to too many permutations error, maybe exclude near ground levels
    fields = ds["W"].isel(level1=25).values

    # print("tracking cells")
    cells = track_cells(
        fields,
        timesteps,
        field_static=field_static,
        min_area=5, # 5
        aura=3,
        threshold=5, # 5
        min_distance=10, # 6
        prominence=5,
        peak_threshold=True,
        min_lifespan = 15,
    )

    # print("gap filling swaths")
    swath = ds["W"].max(dim="time").isel(level1=25).values
    swath_gf = swath
    n_cells = len(cells)
    cell_swaths = np.zeros((n_cells, swath.shape[0], swath.shape[1]))
    cell_ids = []
    for cell in cells:
        swath_gf = np.max(np.dstack((cell.swath, swath_gf)), 2)
        cell_swaths[cell.cell_id, :, :] = cell.swath
        cell_ids.append(cell.cell_id)

    # add empty dimension for time
    swath_gf = np.expand_dims(swath_gf, axis=0)
    swath = np.expand_dims(swath, axis=0)

    swath_gf_nc = xr.Dataset(
        {"W": (["time", "rlat", "rlon"], swath_gf)},
        coords={
            "time": np.atleast_1d(ds["time"][0].values),
            "rlat": ds["rlat"].values,
            "rlon": ds["rlon"].values,
        },
    )
    swath_nc = xr.Dataset(
        {"W": (["time", "rlat", "rlon"], swath)},
        coords={
            "time": np.atleast_1d(ds["time"][0].values),
            "rlat": ds["rlat"].values,
            "rlon": ds["rlon"].values,
        },
    )

    cell_swaths_nc = xr.Dataset(
        {"cell_swath": (["cell_id", "rlat", "rlon"], cell_swaths)},
        coords={
            "cell_id": cell_ids,
            "rlat": ds["rlat"].values,
            "rlon": ds["rlon"].values,
        },
    )
    # add lat lon
    swath_gf_nc["lat"] = xr.DataArray(
        ds["lat"].values,
        dims=["rlat", "rlon"],
        coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
    )
    swath_gf_nc["lon"] = xr.DataArray(
        ds["lon"].values,
        dims=["rlat", "rlon"],
        coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
    )

    cell_swaths_nc["lat"] = xr.DataArray(
        ds["lat"].values,
        dims=["rlat", "rlon"],
        coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
    )
    cell_swaths_nc["lon"] = xr.DataArray(
        ds["lon"].values,
        dims=["rlat", "rlon"],
        coords={"rlat": ds["rlat"].values, "rlon": ds["rlon"].values},
    )

    # add long name to swath
    swath_gf_nc.attrs["long_name"] = "daily gap filled swath using cell tracking"
    # add units to swath
    swath_gf_nc.attrs["units"] = "mm"

    swath_nc.attrs["long_name"] = "daily swath"
    swath_nc.attrs["units"] = "mm"

    # add long name to cell_swaths
    cell_swaths_nc.attrs["long_name"] = (
        "gap filled swath of individual cells using cell tracking"
    )
    # add units to cell_swaths
    cell_swaths_nc.attrs["units"] = "mm"

    with open(os.path.join(outpath, "cells_" + member + ".pickle"), "wb") as f:
        cPickle.dump(cells, f)

    _ = write_to_json(
        cells, os.path.join(outpath, "cell_tracks_" + member + ".json")
    )

    _ = write_masks_to_netcdf(
        cells,
        timesteps,
        field_static,
        os.path.join(outpath, "cell_masks_" + member + ".nc"),
    )

    swath_gf_nc.to_netcdf(
        os.path.join(outpath, "gap_filled_" + member + ".nc"),
        encoding={"W": {"zlib": True, "complevel": 5}},
    )

    swath_nc.to_netcdf(
        os.path.join(outpath, "swath_" + member + ".nc"),
        encoding={"W": {"zlib": True, "complevel": 5}},
    )
    # os.system("nczip " + outfile_gapfilled)
    cell_swaths_nc.to_netcdf(
        os.path.join(outpath, "cell_swaths_" + member + ".nc"),
        encoding={'cell_swath': {'zlib': True, 'complevel': 5}}
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="tracks output files from cosmo-1e hindcasts"
    )
    p.add_argument("inpath", type=str, help="path to input files")
    p.add_argument("outpath", type=str, help="path to output files")

    args = p.parse_args()

    main(**vars(args))
