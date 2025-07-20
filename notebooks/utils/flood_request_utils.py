import json
import os
from matplotlib import pyplot as plt
import requests
import zarr

from scipy.interpolate import interp1d
from fsspec.implementations.local import LocalFileSystem
from physrisk.container import Container
from physrisk.data.inventory_reader import InventoryReader
from plotly.subplots import make_subplots
from IPython.display import Markdown, display


# -----
# Hazard Data
# -----
def get_wri_and_si_hazard_data(coords: dict):
    request = {
        "interpolation": "max",
        "items": [
            {
                "request_item_id": "wri", # friendly name
                "hazard_type": "RiverineInundation",
                "indicator_id": "flood_depth",
                "scenario": "rcp4p5",
                "path": "inundation/wri/v2/inunriver_{scenario}_00000NorESM1-M_{year}",
                "year": 2030,
                "longitudes": [coords["lng"]],
                "latitudes": [coords["lat"]],
            },
            {
                "request_item_id": "si_old", # friendly name
                "hazard_type": "RiverineInundation",
                "indicator_id": "flood_depth",
                "scenario": "historical",
                "path": "inundation/si_poplave/v1/si_poplave_{scenario}_{year}",
                "year": 2025,
                "longitudes": [coords["lng"]],
                "latitudes": [coords["lat"]],
            },
            {
                "request_item_id": "si", # friendly name
                "hazard_type": "RiverineInundation",
                "indicator_id": "flood_depth",
                "scenario": "historical",
                "path": "inundation/si_poplave/v2/si_poplave_{scenario}_{year}",
                "year": 2025,
                "longitudes": [coords["lng"]],
                "latitudes": [coords["lat"]],
            },
        ]
    }

    DATA_DIR = "../data"
    # fdir = os.path.join(DATA_DIR, "test_output_wri_aqueduct2", "hazard", "hazard.zarr")
    fdir = os.path.join(DATA_DIR, "full_models", "hazard", "hazard.zarr")

    # Make sure the directory exists
    if not os.path.exists(fdir):
        raise FileNotFoundError(f"Directory {fdir} does not exist")

    inventory_reader = InventoryReader(fs=LocalFileSystem(), base_path="/Users/klemenkubelj/Documents/school/graduate/masters/code/cvar-masters/data/full_models")
    local_store = zarr.DirectoryStore(fdir)
    container_ls = Container(zarr_store=local_store, inventory_reader=inventory_reader)
    requester_ls = container_ls.requester()

    result = requester_ls.get(request_id="get_hazard_data", request_dict=request)
    data = json.loads(result)
    
    # Parse/clean data for easier processing
    parsed_data = {}
    for item in data["items"]:
        rps = item["intensity_curve_set"][0]["index_values"]
        intensities  = item["intensity_curve_set"][0]["intensities"]
        parsed_data[item["request_item_id"]] = {
            rp: intensity
            for rp, intensity in zip(rps, intensities)
        }
    data["flood_depths"] = parsed_data

    # Apply damage fractions
    processed_damage_fractions = {}
    for k, vals in data["flood_depths"].items():
        processed_damage_fractions[k] = {}
        for rp, depth in vals.items():
            processed_damage_fractions[k][rp] = get_damage_fraction(depth)
    data["damage_fractions"] = processed_damage_fractions

    return data, request


def plot_wri_and_si_hazard_data(data: dict, request: dict, x_axis: str = "RP"):
    """
    Plot the hazard data for the WRI and SI models.

    x_axis can be "RP" (Return Period) or "AEP" (Annual Exceedance Probability).
    """

    if x_axis == "RP":
        x_axis_title = "Return period (years)"
    elif x_axis == "AEP":
        x_axis_title = "Annual exceedance probability"
    else:
        raise ValueError(f"Invalid x_axis: {x_axis}")
    
    fig1 = make_subplots()

    gps = {
        "lat": request["items"][0]["latitudes"][0],
        "lng": request["items"][0]["longitudes"][0],
    }

    for idx, item in enumerate(data["items"]):
        name = request["items"][idx]["path"].format(**request["items"][idx])
        index_values = item["intensity_curve_set"][0]["index_values"]
        if x_axis == "AEP":
            index_values = [1 / rp for rp in index_values]

        print("index_values: ",index_values)
        
        intensities = item["intensity_curve_set"][0]["intensities"]
        print("intensities: ", intensities)
        fig1.add_scatter(x=index_values, y=intensities, name=name, row=1, col=1)

    fig1.update_xaxes(title=x_axis_title, title_font={"size": 14}, row=1, col=1, type="log")
    fig1.update_yaxes(title="Flood depth (m)", title_font={"size": 14}, row=1, col=1)
    fig1.update_layout(legend=dict(orientation="h", y=-0.15))
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig1.update_layout(title=f"Flood depth (m) for different models. GPS: {gps}")
    return fig1


# -----
# Vulnerability Data
# -----
def plot_wri_and_si_vulnerability_data(data: dict, request: dict):
    fig1 = make_subplots()

    gps = {
        "lat": request["items"][0]["latitudes"][0],
        "lng": request["items"][0]["longitudes"][0],
    }

    for idx, item in enumerate(data["items"]):
        name = request["items"][idx]["path"].format(**request["items"][idx])
        index_values = item["intensity_curve_set"][0]["index_values"]
        intensities = item["intensity_curve_set"][0]["intensities"]
        fig1.add_scatter(x=index_values, y=intensities, name=name, row=1, col=1)

    fig1.update_xaxes(title="Return period (years)", title_font={"size": 14}, row=1, col=1, type="log")
    fig1.update_yaxes(title="Relative damage (fraction)", title_font={"size": 14}, row=1, col=1)
    fig1.update_layout(legend=dict(orientation="h", y=-0.15))
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    fig1.update_layout(title=f"Relative damage (fraction) for different models. GPS: {gps}")
    return fig1

def apply_damage_fraction(data: dict):
    risk_data = data.copy()
    for item in risk_data["items"]:
        item["intensity_curve_set"][0]["intensities"] = [get_damage_fraction(depth) for depth in item["intensity_curve_set"][0]["intensities"]]
    return risk_data

def get_damage_fraction(depth: float) -> float:
    """
    Get damage fraction for a given flood depth using polynomial interpolation.
    For depths outside the range, we clamp the values to [0, 1].
    
    Args:
        depth (float): Flood depth in meters
        
    Returns:
        float: Damage fraction between 0 and 1
    """
    # Values taken from JRE Global Flood Damage Estimates (2020)
    flood_depths = [
        0,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        5,
        6
    ]
    # Residential, EU
    damage_fractions = [
        0.00,
        0.25,
        0.40,
        0.50,
        0.60,
        0.75,
        0.85,
        0.95,
        1.00
    ]

    # Create interpolation function
    f = interp1d(flood_depths, damage_fractions, kind="linear", bounds_error=False, fill_value=(0, 1))
    
    
    return float(f(depth))