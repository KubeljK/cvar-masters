from numpy import median
from typing import List
import pandas as pd
import numpy as np


def absolute_diff(real_value: float, predicted_value: float) -> float:
    """
    Given a value and benchmark value, it calculates difference
    as predicted_value - real_value. Difference  can be
    positive or negative.
    """
    return predicted_value - real_value


def relative_diff(real_value: float, predicted_value: float) -> float:
    """
    Given a value and benchmark value, it calculates relative
    difference as relative to benchmark value. Difference  can be
    positive or negative.
    """
    if real_value == 0:
        return 0
    return float((predicted_value - real_value) / real_value)


def within_x(real_values:List[float], predicted_values:List[float], x=20) -> float:
    """
    Given a list of real values and predicted values,
    calculate ratio of predictions within x% of benchmark values.
    Args:
        real_values: list of true values of target feature
        predicted_values: list of predicted values of target feature
        x: integer - error percentage tolerance (i.e. 5, 10, 15, 20)
    """
    n = 0
    for real, pred in zip(real_values, predicted_values):
        # rel_diff = (pred - real) / real
        rel_diff = relative_diff(real, pred)
        if abs(rel_diff) <= (x / 100):
            n += 1
    return n / len(real_values)


def overshoot_x(real_values: List[float], predicted_values: List[float], x: int = 20) -> float:
    """
    Given a list of real values and predicted values,
    calculate share of predictions that overshoot x% of benchmark values.
    Args:
        real_values: list of true values of target feature
        predicted_values: list of predicted values of target feature
        x: integer - error percentage tolerance (i.e. 5, 10, 15, 20)
    """
    n = 0
    for real, pred in zip(real_values, predicted_values):
        # rel_diff = (pred - real) / real
        rel_diff = relative_diff(real, pred)
        if rel_diff >= (x / 100):
            n += 1
    return n / len(real_values)


def median_relative_error(real_values: List[float], predicted_values: List[float]) -> float:
    """
    Given a list of real values and predicted values,
    calculate median relative error.
    Args:
        real_values: list of true values of target feature
        predicted_values: list of predicted values of target feature
    """
    if len(real_values) == 0 or len(predicted_values) == 0:
        return 0

    errors = [relative_diff(real, pred) for real, pred in zip(real_values, predicted_values)]
    return median(errors)


def median_absolute_relative_error(real_values: List[float], predicted_values: List[float]) -> float:
    """
    Given a list of real values and predicted values,
    calculate median absolute relative error.
    Args:
        real_values: list of true values of target feature
        predicted_values: list of predicted values of target feature
    """
    if len(real_values) == 0 or len(predicted_values) == 0:
        return 0

    errors = [abs(relative_diff(real, pred)) for real, pred in zip(real_values, predicted_values)]
    return median(errors)

def st_dev_residuals(real_values: List[float], predicted_values: List[float]) -> float:
    if len(real_values) == 0 or len(predicted_values) == 0:
        return 0

    errors = [pred - real for real, pred in zip(real_values, predicted_values)]
    return np.std(errors, ddof=1)


def get_metrics(real_values: List[float], predicted_values: List[float]) -> dict:
    """
    Given a list of real and predicted values, calculate the most common metrics.
    :param real_values:list
    :param predicted_values:list
    :return: dict
    """
    return {
        # "within_5": within_x(real_values, predicted_values, 5),
        # "within_10": within_x(real_values, predicted_values, 10),
        # "within_15": within_x(real_values, predicted_values, 15),
        "within_20": round(within_x(real_values, predicted_values, 20), 4),
        # "overshoot_20": overshoot_x(real_values, predicted_values, 20),
        # "median_relative_error": median_relative_error(real_values, predicted_values),
        # "median_absolute_relative_error": median_absolute_relative_error(real_values, predicted_values),
        "median_error": round(median([pred - real for real, pred in zip(real_values, predicted_values)]), 4),
        "mean_error": round(np.mean([pred - real for real, pred in zip(real_values, predicted_values)]), 4),
        "median_absolute_error": np.median([abs(pred - real) for real, pred in zip(real_values, predicted_values)]),
        # "mean_absolute_error": np.mean([abs(real - pred) for real, pred in zip(real_values, predicted_values)]),
        "RMSE": round(np.sqrt(np.mean([(pred - real)**2 for real, pred in zip(real_values, predicted_values)])), 4),
        "st_dev_residuals": round(st_dev_residuals(real_values, predicted_values), 4),
        "total_samples": len(real_values),
    }



def get_segmented_metrics(data, att, benchmark_att="benchmark_value", estimated_att="estimated_value"):
    """
    Given a set of data as DataFrame and an attribute,
    create data subsets for for each value of the attribute
    and calculate metrics.

    :param data:
    :param att:
    :return: DataFame
    """
    res = []
    for val in data[att].unique():
        df = data[data[att]==val]
        scores = get_metrics(df[benchmark_att], df[estimated_att])
        scores[att] = val
        res.append(scores)

    return pd.DataFrame(res).sort_values(by=[att])