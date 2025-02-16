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
    scenario: str
    year: str
    filename_return_period: str  # the filename of the input


class SIPoplaveSource:
    def __init__(self, source_dir):
        self.source_dir = source_dir

    # def load_file(self, rp: int) -> gpd.GeoDataFrame:
    #     path = os.path.join(self.source_dir, f"globine_q{rp}.gpkg")
    #     print("Loading file: ", path)
    #     return gpd.read_file(path)

    # def razterize_flood_depth(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    #     pass
    
    @contextmanager
    def load_file(self, fname: str):
        da: Optional[xr.DataArray] = None
        fpath = os.path.join(self.source_dir, fname)
        print("Loading file: ", fpath)
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

    # def open_

    # def prepare(self, working_dir: Optional[str] = None):
    #     path = ""
    #     self.df = gpd.read_file(path)

    # def open_dataset_year(
    #     self, gcm: str, scenario: str, quantity: str, year: int, chunks=None
    # ) -> xr.Dataset:
    #     """_summary_

    #     Args:
    #         gcm (str): Ignored.
    #         scenario (str): Ignored.
    #         quantity (str): 'RiverineInundation' or 'CoastalInundation'.
    #         year (int): Ignored.
    #         chunks (_type_, optional): _description_. Defaults to None.

    #     Returns:
    #         xr.Dataset: Data set named 'indicator' with 'max' and 'min' coordinate labels in the index coordinate.
    #     """
    #     hazard_type = quantity

    #     def get_merged_rp(row, min_max: str, flood_type: str):
    #         """Calculate min or max from database entry (GeoDataFrame row).

    #         From the paper: "In practice, if information is available in the design layer for a given sub-country unit, then
    #         this information is included in the merged layer. If no information is contained in the design layer, then the policy layer
    #         information is included in the merged layer. Finally, if information is not available even at the policy layer, then the
    #         model layer information is included in the merged layer."

    #         Args:
    #             row: GeoDataFrame row
    #             flood_type (str, optional): "Riv" or "Co". Defaults to "Riv".

    #         Returns:
    #             float: Protection level as return period in years.
    #         """
    #         layers = ["DL", "PL", "ModL"] if flood_type == "Riv" else ["DL", "PL"]
    #         for layer in layers:  # design layer, policy layer, modelled layer
    #             # note that for the modelled layer, both min and max are set to the modelled value
    #             layer_rp = (
    #                 row[f"{layer}_{flood_type}"]
    #                 if layer == "ModL"
    #                 else row[f"{layer}_{min_max}_{flood_type}"]
    #             )
    #             if layer_rp > 0:  # if 0, layer is considered missing
    #                 # if design layer is present, use this, otherwise use the policy layer, otherwise the modelled, otherwise missing
    #                 return layer_rp
    #         return float("Nan")  # zero is no data, represented by NaN here.

    #     logger.info(f"Processing hazard type {hazard_type}")
    #     min_shapes: List[Tuple[float, Any]] = []
    #     max_shapes: List[Tuple[float, Any]] = []
    #     logger.info("Inferring max and min protection levels per region")
    #     for _, row in self.df.iterrows():
    #         flood_type = (
    #             "Riv" if hazard_type == "RiverineInundation" else "Co"
    #         )  # riverine and coastal
    #         min, max = (
    #             get_merged_rp(row, "Min", flood_type),
    #             get_merged_rp(row, "Max", flood_type),
    #         )
    #         if row["name"] is None and (min is None and max is None):
    #             continue
    #         # if either the min or max is NaN, that is OK: the vulnerability model is expected to deal with that
    #         if not math.isnan(min) and not math.isnan(max) and min > max:
    #             # it can occur that for a layer there is only information about minimum
    #             raise ValueError("unexpected return period")

    #         if not math.isnan(min):
    #             min_shapes.append((row.geometry, min))
    #         if not math.isnan(max):
    #             max_shapes.append((row.geometry, max))

    #     resolution_in_arc_mins = 1
    #     width, height = (
    #         int(60 * 360 / resolution_in_arc_mins),
    #         int(60 * 180 / resolution_in_arc_mins),
    #     )
    #     crs, transform = global_crs_transform(width, height)
    #     logger.info("Creating empty array")
    #     da = empty_data_array(
    #         width,
    #         height,
    #         transform,
    #         str(crs),
    #         index_name="min_max",
    #         index_values=["min", "max"],
    #     )
    #     for min_max in ["min", "max"]:
    #         logger.info(
    #             f"Creating raster at {(360 * 60) / width} arcmin resolution for {min_max} protection"
    #         )
    #         rasterized = features.rasterize(
    #             min_shapes if min_max == "min" else max_shapes,
    #             out_shape=[height, width],
    #             transform=transform,
    #             all_touched=True,
    #             fill=float("nan"),  # background value
    #             merge_alg=MergeAlg.replace,
    #         )
    #         index = 0 if min_max == "min" else 1
    #         da[index, :, :] = rasterized[:, :]
    #     return da.to_dataset(name="sop")



class SIFloodIndicatorModel(IndicatorModel):
    """On-board the SI Flood model data set from 
    # TODO: Add url for SI Flood model
    """

    def __init__(self):
        pass
        self.return_periods = [
            10, 
            # 100, 
            # 500
        ]

    # def _resource(self, path):
    #     return self.resources[path]

    def batch_items(self) -> Iterable[BatchItem]:
        items = [
            BatchItem(
                path="inundation/si_poplave/v1/si_poplave", # Structur of output path
                scenario="historical", # ?
                year=1971, # ?
                filename_return_period="globine_q{return_period}.tif", # Input filename
            ),
        ]
        return items

    def run_single(
        self, item, source, target: OscZarr, client: Client
    ):
        print("SOURCE: ", source)
        print("TARGET: ", target)
        print("type: ", type(source))
        assert isinstance(source, SIPoplaveSource), "Source is type: " + type(source)
        assert isinstance(target, OscZarr), "Target is type: " + type(target)

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
                        str(CRS.from_epsg(3912)),
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
