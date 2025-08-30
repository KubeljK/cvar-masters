import pandas as pd


def display_metrics_table(metrics_dict, bias_adjusted=False):
    """
    Create a clean table representation of the metrics dictionary.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics for different events and models
        bias_adjusted (bool): Whether the metrics are for bias-adjusted predictions
    
    Returns:
        pd.DataFrame: Formatted table with metrics
    """
    # Check if metrics_dict has the expected structure
    if not metrics_dict:
        raise ValueError("metrics_dict is empty")
    
    # Extract all metric names from the first entry
    first_event = list(metrics_dict.keys())[0]
    
    # Handle different possible structures
    if isinstance(metrics_dict[first_event], dict) and 'si' in metrics_dict[first_event]:
        metric_names = list(metrics_dict[first_event]['si'].keys())
    else:
        raise ValueError(f"Expected metrics structure with 'si' and 'wri' keys, but got: {list(metrics_dict[first_event].keys()) if isinstance(metrics_dict[first_event], dict) else type(metrics_dict[first_event])}")
    
    # Create a list to store all rows
    rows = []
    
    for event in metrics_dict.keys():
        # Add SI row
        si_row = {'Event': event, 'Model': 'SI'}
        for metric in metric_names:
            si_row[metric] = metrics_dict[event]['si'][metric]
        rows.append(si_row)
        
        # Add WRI row
        wri_row = {'Event': event, 'Model': 'WRI'}
        for metric in metric_names:
            wri_row[metric] = metrics_dict[event]['wri'][metric]
        rows.append(wri_row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns to have Event and Model first
    columns = ['Event', 'Model'] + metric_names
    df = df[columns]
    
    return df


def display_metrics_comparison_table(metrics_dict, bias_adjusted=False):
    """
    Create a side-by-side comparison table of SI vs WRI metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics for different events and models
        bias_adjusted (bool): Whether the metrics are for bias-adjusted predictions
    
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    # Extract all metric names from the first entry
    first_event = list(metrics_dict.keys())[0]
    metric_names = list(metrics_dict[first_event]['si'].keys())
    
    # Create a list to store all rows
    rows = []
    
    for event in metrics_dict.keys():
        row = {'Event': event}
        
        # Add metrics for both models
        for metric in metric_names:
            row[f'{metric}_SI'] = metrics_dict[event]['si'][metric]
            row[f'{metric}_WRI'] = metrics_dict[event]['wri'][metric]
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df


def print_metrics_summary(metrics_dict, bias_adjusted=False):
    """
    Print a nicely formatted summary of the metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics for different events and models
        bias_adjusted (bool): Whether the metrics are for bias-adjusted predictions
    """
    adjustment_text = " (Bias Adjusted)" if bias_adjusted else ""
    print(f"FLOOD DEPTH PREDICTION METRICS{adjustment_text}")
    print("=" * 60)
    
    # Key metrics to highlight
    key_metrics = ['within_20', 'mean_error', 'RMSE', 'st_dev_residuals', 'total_samples']
    
    for event in metrics_dict.keys():
        print(f"\n{event}:")
        print("-" * 30)
        print(f"{'Metric':<20} {'SI':<12} {'WRI':<12}")
        print("-" * 44)
        
        for metric in key_metrics:
            si_val = metrics_dict[event]['si'][metric]
            wri_val = metrics_dict[event]['wri'][metric]
            
            # Format values based on metric type
            if metric == 'within_20':
                si_str = f"{si_val:.1%}"
                wri_str = f"{wri_val:.1%}"
            elif metric == 'total_samples':
                si_str = f"{si_val:,}"
                wri_str = f"{wri_val:,}"
            else:
                si_str = f"{si_val:.4f}"
                wri_str = f"{wri_val:.4f}"
            
            print(f"{metric:<20} {si_str:<12} {wri_str:<12}")


def create_latex_table(metrics_dict, bias_adjusted=False):
    """
    Create a LaTeX table representation of the metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metrics for different events and models
        bias_adjusted (bool): Whether the metrics are for bias-adjusted predictions
    
    Returns:
        str: LaTeX table code
    """
    adjustment_text = " (Bias Adjusted)" if bias_adjusted else ""
    
    latex = []
    latex.append("\\begin{table}[htb]")
    latex.append("  \\centering")
    latex.append(f"  \\caption{{Flood depth prediction quality metrics{adjustment_text.lower()}.}}")
    latex.append("  \\label{tab:metrics}")
    latex.append("  \\begin{tabular}{|l|l|r|r|r|r|r|}")
    latex.append("    \\hline")
    latex.append("    \\textbf{Event} & \\textbf{Model} & \\textbf{Within 20\\%} & \\textbf{Mean Error} & \\textbf{RMSE} & \\textbf{Std Dev} & \\textbf{Samples} \\\\")
    latex.append("    \\hline")
    
    for event in metrics_dict.keys():
        # SI row
        si_metrics = metrics_dict[event]['si']
        latex.append(f"    {event} & SI & {si_metrics['within_20']:.1%} & {si_metrics['mean_error']:.4f} & {si_metrics['RMSE']:.4f} & {si_metrics['st_dev_residuals']:.4f} & {si_metrics['total_samples']:,} \\\\")
        
        # WRI row
        wri_metrics = metrics_dict[event]['wri']
        latex.append(f"    & WRI & {wri_metrics['within_20']:.1%} & {wri_metrics['mean_error']:.4f} & {wri_metrics['RMSE']:.4f} & {wri_metrics['st_dev_residuals']:.4f} & {wri_metrics['total_samples']:,} \\\\")
        latex.append("    \\hline")
    
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)
