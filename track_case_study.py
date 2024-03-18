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

    # iterate over member (todo: paralelize)
    for member in members:
        print("processing member: ", member)

        # load data
        memberpath = os.path.join(inpath, member)

        ds = xr.open_mfdataset(os.path.join(memberpath, "lfff*.nc"), combine="by_coords")

        field_static = {}
        field_static["lat"] = ds["lat"].values
        field_static["lon"] = ds["lon"].values

        timesteps = ds["time"].values

        fields = ds["W"].max(dim='level1').values

        print("tracking cells")
        cells = track_cells(
            fields, timesteps, field_static=field_static, min_area=5, aura=3
        )

        with open(os.path.join(outpath, "cells_" + member + ".pickle"), "wb") as f:
            cPickle.dump(cells, f)

        _ = write_to_json(cells, os.path.join(outpath, "cell_tracks_" + member + ".json"))

        _ = write_masks_to_netcdf(cells, timesteps, field_static, os.path.join(outpath, "cell_masks_" + member + ".nc"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="tracks output files from cosmo-1e hindcasts"
    )
    p.add_argument("inpath", type=str, help="path to input files")
    p.add_argument("outpath", type=str, help="path to output files")

    args = p.parse_args()

    main(**vars(args))
