import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

from affine import Affine
from dask.distributed import Client
from pydantic import parse_obj_as  # type: ignore

from hazard.indicator_model import IndicatorModel
from hazard.inventory import HazardResource, Period
from hazard.protocols import WriteDataArray
from hazard.sources.osc_zarr import OscZarr
from hazard.sources.wri_aqueduct import WRIAqueductSource
from hazard.utilities.map_utilities import alphanumeric
from hazard.utilities.tiles import create_tile_set

from contextlib import contextmanager
import xarray as xr
from rasterio.crs import CRS  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    path: str
    filename_return_period: str  # the filename of the input


class SIPoplaveSource:
    def __init__(self, source_dir):
        self.source_dir = source_dir
    
    @contextmanager
    def load_file(self, fname: str):
        da: Optional[xr.DataArray] = None
        fpath = os.path.join(self.source_dir, fname)
        f = None
        try:
            f = open(fpath, "rb")
            da = xr.open_rasterio(f)
            yield da
        finally:
            if da is not None:
                da.close()
            if f is not None:
                f.close()


class SIFloodIndicatorModel(IndicatorModel):
    """On-board the SI Flood model data set from 
    # TODO: Add url for SI Flood model
    """

    def __init__(self, epsg: int = 3912):
        self.return_periods = [
            # 10,
            100,
            # 500,
        ]
        self.grid_sizes = [
            10,
            100,
            1000,
        ]
        self.epsg = epsg

    def batch_items(self) -> Iterable[BatchItem]:
        items = []
        for grid_size in self.grid_sizes:
            path = f"inundation/si_poplave/v1/si_poplave_historical_2025_{grid_size}"
            if grid_size == 10:
                path = path.replace("_10", "")
            items.append(
                BatchItem(
                    path=path, # Structure of output path
                    filename_return_period=f"globine_q{{return_period}}_{grid_size}.tif", # Input filename
                )
            )
        return items

    def run_single(
        self, item, source, target: OscZarr, client: Client
    ):
        print("SOURCE: ", source)
        print("TARGET: ", target)
        print("type: ", type(source))
        # assert isinstance(source, SIPoplaveSource), "Source is type: " + str(type(source)) + ". Expected type: " + str(SIPoplaveSource)
        # assert isinstance(target, OscZarr), "Target is type: " + str(type(target)) + ". Expected type: " + str(OscZarr)
        
        epsg_3912 = """
        PROJCS["MGI_1901_Slovene_National_Grid",GEOGCS["GCS_MGI_1901",DATUM["D_MGI_1901",SPHEROID["Bessel_1841",6377397.155,299.1528128]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",-5000000.0],PARAMETER["Central_Meridian",15.0],PARAMETER["Scale_Factor",0.9999],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]
        """.replace("\n", "")

        epsg_3974 = """
        PROJCS["Slovenia 1996 / Slovene National Grid",GEOGCS["Slovenia 1996",DATUM["Slovenia_Geodetic_Datum_1996",SPHEROID["GRS 1980",6378137,298.257222101],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4765"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",500000],PARAMETER["false_northing",-5000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3794"]]
        """

        if self.epsg == 3912:
            epsg_str = epsg_3912
        elif self.epsg == 3974:
            epsg_str = epsg_3974
        else:
            raise ValueError(f"Invalid EPSG code: {self.epsg}")

        for i, ret in enumerate(self.return_periods):
            print(f"Copying return period {i + 1}/{len(self.return_periods)}")
            fname = item.filename_return_period.format(return_period=ret)
            print("Loading fname: ", fname)

            with source.load_file(fname) as da:
                assert da is not None
                if ret == self.return_periods[0]:
                    z = target.create_empty(
                        item.path,
                        len(da.x),
                        len(da.y),
                        Affine(
                            da.transform[0],
                            da.transform[1],
                            da.transform[2],
                            da.transform[3],
                            da.transform[4],
                            da.transform[5],
                        ),
                        epsg_str,
                        index_values=self.return_periods,
                    )
                # ('band', 'y', 'x')
                values = da[
                    0, :, :
                ].data  # will load into memory; assume source not chunked efficiently
                values[values == -9999.0] = float("nan")
                z[i, :, :] = values
        print("done")

    def inventory(self) -> Iterable[HazardResource]:
        return []
