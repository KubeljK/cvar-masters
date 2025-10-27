import json
import os
from matplotlib import pyplot as plt
import requests
import zarr
import numpy as np

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
            {
                "request_item_id": "si_res_100", # friendly name
                "hazard_type": "RiverineInundation",
                "indicator_id": "flood_depth",
                "scenario": "historical",
                "path": "inundation/si_poplave/v2/si_poplave_{scenario}_{year}_100",
                "year": 2025,
                "longitudes": [coords["lng"]],
                "latitudes": [coords["lat"]],
            },
            {
                "request_item_id": "si_res_1000", # friendly name
                "hazard_type": "RiverineInundation",
                "indicator_id": "flood_depth",
                "scenario": "historical",
                "path": "inundation/si_poplave/v2/si_poplave_{scenario}_{year}_1000",
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


def plot_wri_and_si_hazard_data(
        data: dict,
        request: dict,
        x_axis: str = "RP",
        plot_from: list = ["si_old", "wri"],
        logscale: bool = True,
        show_legend: bool = True,
        legend_loc: str = "upper left"
    ):
    """
    Plot the hazard data for the WRI and SI models using matplotlib.

    x_axis can be "RP" (Return Period) or "AEP" (Annual Exceedance Probability).
    """

    if x_axis == "RP":
        x_axis_title = "Return period"
    elif x_axis == "AEP":
        x_axis_title = "Average exceedance probability"
    else:
        raise ValueError(f"Invalid x_axis: {x_axis}")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    gps = {
        "lat": request["items"][0]["latitudes"][0],
        "lng": request["items"][0]["longitudes"][0],
    }
    lat_str = f"{gps['lat']:.5f}"
    lng_str = f"{gps['lng']:.5f}"

    for idx, item in enumerate(data["items"]):
        # Friendly, compact legend labels
        req_item = request["items"][idx]
        rid = req_item.get("request_item_id", "")
        if rid not in plot_from:
            continue
        friendly = {
            "wri": "WRI Aqueduct",
            "si_old": "SI IKG v1",
            "si": "SI v2 - 10m resolution",
            "si_res_100": "SI v2 - 100m resolution",
            "si_res_1000": "SI v2 - 1000m resolution",
        }.get(rid, rid or "Model")
        scenario = req_item.get("scenario", "?")
        year = req_item.get("year", "?")
        name = f"{friendly}"# ({scenario} {year})"
        index_values = item["intensity_curve_set"][0]["index_values"]
        if x_axis == "AEP":
            index_values = [1 / rp for rp in index_values]

        intensities = item["intensity_curve_set"][0]["intensities"]
        ax.plot(index_values, intensities, marker="o", markersize=4, linewidth=1.6, label=name)

    if logscale:
        ax.set_xscale("log")
    ax.set_xlabel(x_axis_title, fontsize=14)
    ax.set_ylabel("Flood depth [m]", fontsize=14)
    # ax.set_title(f"Flood depth (m) for different models (lat={lat_str}, lng={lng_str})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(x=0.05)

    # ax.set_ylim(0, None)

    # Place legend outside the plot on the right to avoid crowding
    fig.subplots_adjust(right=0.75)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=14)
    fig.tight_layout()

    return fig


# -----
# Vulnerability Data
# -----
residential_damage_fractions = [
    (0, 0.00),
    (0.5, 0.25),
    (1, 0.40),
    (1.5, 0.50),
    (2, 0.60),
    # (2.5, 0.675),
    (3, 0.75),
    (4, 0.85),
    (5, 0.95),
    (6, 1.00),
]
residential_damage_fractions_calibrated = [
    (0, 0.00),
    (0.5, 0.3),
    (1, 0.50),
    (1.5, 0.50),
    (2, 0.50),
    # (2.5, 0.675),
    (3, 0.75),
    (4, 0.85),
    (5, 0.95),
    (6, 1.00),
]
commercial_damage_function = [
    (0, 0.00),
    (0.5, 0.15),
    (1, 0.30),
    (1.5, 0.45),
    (2, 0.55),
    # (2.5, 0.675),
    (3, 0.75),
    (4, 0.90),
    (5, 1.00),
    (6, 1.00),
]
commercial_damage_function_calibrated = [
    (0, 0.00),
    (0.5, 0.3),
    (1, 0.50),
    (1.5, 0.8),
    (2, 0.8),
    # (2.5, 0.675),
    (3, 0.75),
    (4, 0.90),
    (5, 1.00),
    (6, 1.00),
]
industrial_damage_function = [
    (0, 0.00),
    (0.5, 0.15),
    (1, 0.27),
    (1.5, 0.40),
    (2, 0.52),
    # (2.5, 0.64),
    (3, 0.70),
    (4, 0.85),
    (5, 1.00),
    (6, 1.00),
]
industrial_damage_function_calibrated = [
    (0, 0.00),
    (0.5, 0.3),
    (1, 0.425),
    (1.5, 0.5),
    (2, 0.65),
    # (2.5, 0.64),
    (3, 0.70),
    (4, 0.85),
    (5, 1.00),
    (6, 1.00),
]
agriculture_damage_function = [
    (0, 0.00),
    (0.5, 0.30),
    (1, 0.55),
    (1.5, 0.65),
    (2, 0.75),
    # (2.5, 0.80),
    (3, 0.85),
    (4, 0.95),
    (5, 1.00),
    (6, 1.00),
]
agriculture_damage_function_calibrated = [
    (0, 0.00),
    (0.5, 0.30),
    (1, 0.3),
    (1.5, 0.5),
    (2, 0.5),
    # (2.5, 0.80),
    (3, 0.85),
    (4, 0.95),
    (5, 1.00),
    (6, 1.00),
]

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

def get_depth_damage_function(property_type: str):
    _map = {
        "residential": residential_damage_fractions,
        "commercial": commercial_damage_function,
        "industrial": industrial_damage_function,
        "agriculture": agriculture_damage_function,
    }
    # Create interpolation function
    damage_function = _map[property_type]
    return damage_function

def get_damage_fraction(depth: float, property_type: str = None) -> float:
    """
    Get damage fraction for a given flood depth using polynomial interpolation.
    For depths outside the range, we clamp the values to [0, 1].
    
    Args:
        depth (float): Flood depth in meters
        
    Returns:
        float: Damage fraction between 0 and 1
    """
    # Values taken from JRE Global Flood Damage Estimates (2020)
    if not property_type:
        # print("Property type not provided, using residential as default!!!!!!")
        property_type = "residential"
    
    damage_function = get_depth_damage_function(property_type)
    f = interp1d([i[0] for i in damage_function], [i[1] for i in damage_function], kind="linear", bounds_error=False, fill_value=(0, 1))
    return float(f(depth))

def plot_damage_function_full_range(property_type: str, color="orange", label="NONE"):
    fig = plt.gcf()
    ax = plt.gca()
    damage_function = get_depth_damage_function(property_type)

    flood_depths = [i[0] for i in damage_function]
    damage_fractions = [i[1] for i in damage_function]

    # Create points for smooth curve visualization
    depths_smooth = np.linspace(0, max(flood_depths), 100)
    damage_smooth = [get_damage_fraction(d, property_type) for d in depths_smooth]

    # Plot the data and fitted curve
    ax.plot(flood_depths, damage_fractions, "o", color=color)
    # Dont show this in legend
    ax.plot(depths_smooth, damage_smooth, "-", label=label, color=color)
    
    # Set y-axis range from 0 to 1
    # ax.set_ylim(0, 1)
    
    # Add labels and title
    ax.set_xlabel("Flood depth [m]")
    ax.set_ylabel("Damage")
    # ax.set_title("Damage Function")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Return the figure and axis for further customization
    return fig, ax