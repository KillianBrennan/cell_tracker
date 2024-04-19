"""
tracks output files from the climate simulations and saves daily (06-06 utc) cell tracks and masks
takes daily aggregated files as input

input:
    inpath: path to input files
    outpath: path to output files
    start_day: start day YYYYMMDD
    end_day: end day YYYYMMDD

output:
    cell_tracks_YYYYMMDD.json: json file containing cell tracks
    cell_masks_YYYYMMDD.nc: netcdf file containing cell masks

example use:
python /home/kbrennan/cell_tracker/track_climate_precip.py /home/kbrennan/phd/data/climate/present/5min_2D/ /home/kbrennan/phd/data/climate/tracks/present/test_precip 20210620 20210620

with memory profiling:
python -m memory_profiler -o /home/kbrennan/cell_tracker/20210620_precip_mbenchmark.prof /home/kbrennan/cell_tracker/track_climate_precip.py /home/kbrennan/phd/data/climate/present/5min_2D/ /home/kbrennan/phd/data/climate/tracks/present/test_precip 20210620 20210620

with compute profiling:
python -m cProfile -o /home/kbrennan/cell_tracker/20210620_precip_cbenchmark.prof /home/kbrennan/cell_tracker/track_climate_precip.py /home/kbrennan/phd/data/climate/present/5min_2D/ /home/kbrennan/phd/data/climate/tracks/present/test_precip 20210620 20210620

"""

import os
import argparse

import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt

import _pickle as cPickle

from cell_tracker import (
    Cell,
    track_cells,
    write_to_json,
    write_masks_to_netcdf,
)


def main(inpath, outpath, start_day, end_day):
    # set tracking parameters
    tracking_parameters = {
        "prominence": 8.2 / 12,  # (I convert 8.2 mm/h -> 8.2/12 kg/m^2/5min)
        "threshold": 5.5 / 12,
        "min_distance": 5,
        "fill_method": "watershed",
        "advection_method": "movement_vector",
        "dynamic_tracking": 3,
        "v_limit": 5,
        "min_area": 24,
        "alpha": 0.5,
        "min_lifespan": 30,
        "aura": 1,
        "quiet": False,
        "cluster_size_limit": 12,
        "sparse_memory": True,
    }

    start_day = pd.to_datetime(start_day, format="%Y-%m-%dT%H")
    end_day = pd.to_datetime(end_day, format="%Y-%m-%dT%H")

    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)

    # iterate over each day in the dataset
    for i, day in enumerate(daylist):
        # print progress
        print("processing day ", i + 1, " of ", len(daylist))
        print("processing day: ", day.strftime("%Y%m%d"))
        # get the data for one day
        print("loading data")

        path = os.path.join(inpath, "lffd" + day.strftime("%Y%m%d") + "_0606.nz")

        ds = xr.open_dataset(path, engine="netcdf4")

        # round minutes to 5
        ds["time"] = ds["time"].dt.round("5min")

        field_static = {}
        field_static["lat"] = ds["lat"].values
        field_static["lon"] = ds["lon"].values

        timesteps = ds["time"].values

        fields = ds["TOT_PREC"].values

        # track cells
        print("tracking cells")
        cells = track_cells(
            fields,
            timesteps,
            field_static=field_static,
            **tracking_parameters,
        )

        print("writing data to file")
        outfile_json = os.path.join(
            outpath, "cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        )
        outfile_nc = os.path.join(
            outpath, "cell_masks_" + day.strftime("%Y%m%d") + ".nc"
        )

        _ = write_to_json(cells, outfile_json)

        _ = write_masks_to_netcdf(
            cells,
            timesteps,
            field_static,
            outfile_nc,
            include_lat_lon=False,
        )

    print("finished tracking all days in queue")
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="tracks cells")
    p.add_argument("inpath", type=str, help="path to input files")
    p.add_argument("outpath", type=str, help="path to output files")
    p.add_argument("start_day", type=str, help="start day")
    p.add_argument("end_day", type=str, help="end day")
    args = p.parse_args()

    main(**vars(args))
