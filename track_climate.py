"""
tracks output files from the climate simulations and saves daily (06-06 utc) cell tracks and masks
takes daily aggregated files as input

input:
    inpath: path to input files
    outpath: path to output files
    start_day: start day
    end_day: end day

output:
    cell_tracks_YYYYMMDD.json: json file containing cell tracks
    cell_masks_YYYYMMDD.nc: netcdf file containing cell masks


example use

python /home/kbrennan/phd/scripts/tracking/track_climate.py /home/kbrennan/phd/data/climate/cosmo6_2017/data /home/kbrennan/phd/data/climate/cosmo6_2017/tracks 20170601 20170605

"""
import sys
import os
import argparse

import xarray as xr
import pandas as pd
import numpy as np

sys.path.insert(1, "/home/kbrennan/phd/scripts")

from tracking.track_cells_v2 import (
    Cell,
    track_cells,
    write_to_json,
    write_masks_to_netcdf,
)


def main(inpath, outpath, start_day, end_day):
    # set tracking parameters
    min_distance = 5
    dynamic_tracking = 4
    v_limit = 5
    min_area = 16
    quiet = False

    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")

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

        field_static = {}
        field_static["lat"] = ds["lat"].values
        field_static["lon"] = ds["lon"].values

        timesteps = ds["time"].values

        fields = ds["DHAIL_MX"].values

        # track cells
        print("tracking cells")
        cells = track_cells(
            fields,
            timesteps,
            field_static=field_static,
            quiet=quiet,
            min_distance=min_distance,
            dynamic_tracking=dynamic_tracking,
            v_limit=v_limit,
            min_area=min_area,
        )

        print("gap filling swaths")
        swath = ds["DHAIL_MX"].max(dim="time").values
        swath_gf = swath
        for cell in cells:
            swath_gf = np.max(np.dstack((cell.swath, swath_gf)), 2)

        # add empty dimension for time
        swath_gf = np.expand_dims(swath_gf, axis=0)

        swath_gf_nc = xr.Dataset(
            {"DHAIL_MX": (["time", "rlat", "rlon"], swath_gf)},
            coords={
                "time": np.atleast_1d(ds["time"][0].values),
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

        # add long name to swath
        swath_gf_nc.attrs["long_name"] = "gap filled swath using cell tracking"
        # add units to swath
        swath_gf_nc.attrs["units"] = "mm"

        print("writing data to file")
        outfile_json = os.path.join(
            outpath, "cell_tracks_" + day.strftime("%Y%m%d") + ".json"
        )
        outfile_nc = os.path.join(
            outpath, "cell_masks_" + day.strftime("%Y%m%d") + ".nc"
        )

        outfile_gapfilled = os.path.join(
            outpath, "gap_filled_" + day.strftime("%Y%m%d") + ".nc"
        )
        _ = write_to_json(cells, outfile_json)

        _ = write_masks_to_netcdf(cells, timesteps, field_static, outfile_nc)

        swath_gf_nc.to_netcdf(outfile_gapfilled)
        os.system("nczip " + outfile_gapfilled)

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
