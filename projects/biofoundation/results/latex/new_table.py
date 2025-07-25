import pandas as pd
import numpy as np
import os
import sys

import numpy


def calculate_mean_std(values):
    """Calculate mean and std from a list of values and return as separate values"""
    if not values or all(v == 0 for v in values):
        return None, None
    
    valid_values = [v for v in values if v > 0]
    if not valid_values:
        return None, None
    
    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0
    
    return mean_val, std_val


def format_mean_std_display(mean_val, std_val):
    """Format mean and std for display as MM.mm\\scriptsize{$\\pm$ std}"""
    if mean_val is None or std_val is None:
        return "-"
    return f"{mean_val:.2f}\\scriptsize{{$\\pm$ {std_val:.2f}}}"


# === Functions ===
# Format row with LaTeX
def format_values(values):
    # Transpose the list of lists to treat each position as a column
    columns = np.array(values, dtype=object).T
    formatted_columns = []

    for col in columns:
        # Extract means from the values (tuples or formatted strings)
        means = []
        processed_values = []
        
        for val in col:
            if isinstance(val, tuple) and len(val) == 2:
                # It's a (mean, std) tuple
                mean_val, std_val = val
                if mean_val is not None:
                    means.append(mean_val)
                    processed_values.append(format_mean_std_display(mean_val, std_val))
                else:
                    means.append(0)
                    processed_values.append("-")
            elif isinstance(val, str) and "$\\pm$" in val:
                # Handle both old format: MM.mm($\pm$std) and new format: MM.mm\scriptsize{$\pm$ std}
                if "($\\pm$" in val:
                    mean_part = val.split("($\\pm$")[0]
                elif "\\scriptsize{$\\pm$" in val:
                    mean_part = val.split("\\scriptsize{$\\pm$")[0]
                else:
                    mean_part = val.split("$\\pm$")[0]
                means.append(float(mean_part))
                processed_values.append(val)
            elif isinstance(val, str) and val != "-":
                try:
                    means.append(float(val))
                    processed_values.append(val)
                except ValueError:
                    means.append(0)
                    processed_values.append("-")
            else:
                means.append(0)
                processed_values.append("-")
        
        means = np.array(means)
        max_idx = np.argmax(means)
        second_max_idx = np.argsort(means)[-2] if len(means) > 1 else 0
        
        formatted = []
        for i, val in enumerate(processed_values):
            if val == "-":
                formatted.append("-")
            elif i == max_idx:
                formatted.append(f"\\textbf{{{val}}}")
            elif i == second_max_idx:
                formatted.append(f"\\underline{{{val}}}")
            else:
                formatted.append(str(val))
        formatted_columns.append(formatted)

    # Transpose back to match the original structure
    return np.array(formatted_columns, dtype=object).T.tolist()


def format_values_no_bold(values):  # This just handles values for one model
    # values should be tuples (mean, std) or already processed numbers/strings
    if isinstance(values, list):
        formatted = []
        for val in values:
            if isinstance(val, tuple) and len(val) == 2:
                # It's a (mean, std) tuple
                mean_val, std_val = val
                if mean_val is not None:
                    formatted.append(format_mean_std_display(mean_val, std_val))
                else:
                    formatted.append("-")
            elif isinstance(val, str):
                formatted.append(val)
            elif val > 0:
                formatted.append(f"{val:.1f}")
            else:
                formatted.append("-")
        return formatted
    else:
        # Handle single value
        if isinstance(values, tuple) and len(values) == 2:
            mean_val, std_val = values
            if mean_val is not None:
                return format_mean_std_display(mean_val, std_val)
            else:
                return "-"
        else:
            rounded = np.round(values, 1)
            formatted = [f"{val:.1f}" if val > 0 else "-" for val in rounded]
            return formatted


def format_name(name):
    if "-" in name:
        parts = name.split("-")
        name = (
            "\\rotatebox[origin=c]{90}{\\renewcommand{\\arraystretch}{1}\\begin{tabular}{@{}c@{}}"
            + " \\\\ ".join(parts)
            + "\\end{tabular}}\n"
        )
    else:
        name = "\\rotatebox[origin=c]{90}{" + name + "}\n"
    return name


def format_hm(values, color):
    # Handle both tuples (mean, std) and formatted strings
    means = []
    processed_values = []
    
    for val in values:
        if isinstance(val, tuple) and len(val) == 2:
            # It's a (mean, std) tuple
            mean_val, std_val = val
            if mean_val is not None:
                means.append(mean_val)
                processed_values.append((mean_val, std_val))
            else:
                means.append(0)
                processed_values.append("-")
        elif isinstance(val, str) and "$\\pm$" in val:
            # Handle both old format: MM.mm($\pm$std) and new format: MM.mm\scriptsize{$\pm$ std}
            if "($\\pm$" in val:
                mean_part = val.split("($\\pm$")[0]
            elif "\\scriptsize{$\\pm$" in val:
                mean_part = val.split("\\scriptsize{$\\pm$")[0]
            else:
                mean_part = val.split("$\\pm$")[0]
            means.append(float(mean_part))
            processed_values.append(val)
        elif isinstance(val, str) and val != "-":
            try:
                means.append(float(val))
                processed_values.append(val)
            except ValueError:
                means.append(0)
                processed_values.append("-")
        else:
            means.append(0 if val != "-" else 0)
            processed_values.append("-")
    
    means = np.array(means)
    max_idx = np.argmax(means)
    second_max_idx = np.argsort(means)[-2] if len(means) > 1 else 0
    
    formatted = []
    for i, val in enumerate(processed_values):
        if val == "-":
            formatted.append("-")
        elif isinstance(val, tuple):
            mean_val, std_val = val
            if mean_val is None:
                formatted.append("-")
            elif i == max_idx:
                formatted.append(f"\\heat{color}[bold]{{{mean_val:.2f}}}")
            elif i == second_max_idx:
                formatted.append(f"\\heat{color}[underline]{{{mean_val:.2f}}}")
            else:
                formatted.append(f"\\heat{color}{{{mean_val:.2f}}}")
        else:
            # Handle legacy formatted strings
            if i == max_idx:
                formatted.append(f"\\heat{color}[bold]{{\\text{{{val}}}}}")
            elif i == second_max_idx:
                formatted.append(f"\\heat{color}[underline]{{\\text{{{val}}}}}")
            else:
                formatted.append(f"\\heat{color}{{\\text{{{val}}}}}")
    return formatted


def format_hm_no_bold(values, color):  # This just handles values for one model
    formatted = []
    for val in values:
        if isinstance(val, tuple) and len(val) == 2:
            # It's a (mean, std) tuple
            mean_val, std_val = val
            if mean_val is None:
                formatted.append("-")
            else:
                formatted.append(f"\\heat{color}{{{mean_val:.2f}}}")
        elif isinstance(val, str):
            if val == "-" or val == "0":
                formatted.append("-")
            else:
                # Legacy formatted string - protect from PGF parsing
                formatted.append(f"\\heat{color}{{\\text{{{val}}}}}")
        elif val > 0:
            formatted.append(f"\\heat{color}{{{val:.1f}}}")
        else:
            formatted.append("-")
    return formatted


# === BEANS === (Adjusted for one table so it returns the needed lists)
def beans_table(path, models, restricted, auroc):
    df = pd.read_csv(path, sep=",")
    metric = "Top1" if not auroc else "auroc"

    # Rename for convenience
    df = df.rename(
        columns={
            "datamodule.dataset.dataset_name": "Dataset",
            "module.network.model_name": "Model",
            "tags": "Tags",
            "module.network.model.pooling": "Pooling",
            "test/MulticlassAccuracy": "Top1",
            "test/AUROC": "auroc",
            "module.network.model.restrict_logits": "Restrict",
        }
    )

    # if the "restrict" is present in the tags, set Restrict to True
    # df["Restrict"] = df["Tags"].str.contains("restrict", case=False)

    # Convert scores to percentage
    df[metric] *= 100

    datasets = [
        "beans_watkins",
        "beans_bats",
        "beans_cbi",
        "beans_dogs",
        "beans_humbugdb",
    ]  # This will not change often

    # === Build table rows ===
    # Initialize lists to store all values for later processing
    all_top1_lp, all_top1_ft, all_top1_ap = [], [], []
    all_avg_top1_lp, all_avg_top1_ft, all_avg_top1_ap = [], [], []
    res_results = {}

    print(f"Processing {len(models)} models for beans table...")

    # Collect data for all models
    for model in models:
        top1_lp, top1_ft, top1_ap = [], [], []

        for dataset in datasets:
            lp_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("linearprobing"))
                & (df["Pooling"] != "attentive")
                & (df["Pooling"] != "average")
                & (df["Restrict"] != True)
            ]
            ft_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("finetune|finetuning"))
                & (df["Pooling"] != "attentive")
                & (df["Pooling"] != "average")
                & (df["Restrict"] != True)
            ]

            ap_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Pooling"] == "attentive")
                & (df["Restrict"] != True)
            ]
            # If restricted, calculate the results for the restricted model for CBI and save in a dictionary
            if (
                restricted
                and dataset == "beans_cbi"
                and (model == "surfperch" or model == "perch" or model == "convnext_bs")
            ):
                # Calculate restricted results in isolated form for easy removal
                res_rows = df[
                    (df["Model"] == model)
                    & (df["Dataset"] == dataset)
                    & (df["Restrict"] == True)
                ]
                top1_res_values = res_rows[metric].tolist() if not res_rows.empty else []
                top1_res = calculate_mean_std(top1_res_values)
                avg_top1_res = top1_res
                # Do the formatting separately
                top1_res = format_values_no_bold([top1_res])[0]
                avg_top1_res = format_hm_no_bold([avg_top1_res], "blue")[0]

                res_results[model] = {"top1": top1_res, "avg_top1": avg_top1_res}

            # Calculate mean and std instead of max
            lp_values = lp_rows[metric].tolist() if not lp_rows.empty else []
            ft_values = ft_rows[metric].tolist() if not ft_rows.empty else []
            ap_values = ap_rows[metric].tolist() if not ap_rows.empty else []
            
            top1_lp.append(calculate_mean_std(lp_values))
            top1_ft.append(calculate_mean_std(ft_values))
            top1_ap.append(calculate_mean_std(ap_values))
        # Averages - calculate from the mean values extracted from tuples
        lp_means = []
        ft_means = []
        ap_means = []
        
        for val in top1_lp:
            if isinstance(val, tuple) and len(val) == 2 and val[0] is not None:
                lp_means.append(val[0])  # Extract mean from tuple
        for val in top1_ft:
            if isinstance(val, tuple) and len(val) == 2 and val[0] is not None:
                ft_means.append(val[0])  # Extract mean from tuple
        for val in top1_ap:
            if isinstance(val, tuple) and len(val) == 2 and val[0] is not None:
                ap_means.append(val[0])  # Extract mean from tuple
        
        avg_top1_lp = calculate_mean_std(lp_means)
        avg_top1_ft = calculate_mean_std(ft_means)
        avg_top1_ap = calculate_mean_std(ap_means)

        # Store all values for later processing
        all_top1_lp.append(top1_lp)
        all_top1_ft.append(top1_ft)
        all_top1_ap.append(top1_ap)
        all_avg_top1_lp.append(avg_top1_lp)
        all_avg_top1_ft.append(avg_top1_ft)
        all_avg_top1_ap.append(avg_top1_ap)

    print(f"Collected data for {len(all_top1_lp)} models in beans table")

    # Determine the highest and second highest values and write LaTeX
    # Instead of using format_values which transposes, we need a different approach for tuples
    print("Formatting LP data...")
    
    # Convert tuples to formatted strings first
    processed_lp = []
    for model_data in all_top1_lp:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_lp.append(model_row)
    
    # Now apply the formatting for bold/underline
    all_top1_lp = format_values(processed_lp)
    
    # Do the same for FT and AP data
    processed_ft = []
    for model_data in all_top1_ft:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_ft.append(model_row)
    all_top1_ft = format_values(processed_ft)
    
    processed_ap = []
    for model_data in all_top1_ap:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_ap.append(model_row)
    all_top1_ap = format_values(processed_ap)

    all_avg_top1_lp = format_hm(all_avg_top1_lp, "green")
    all_avg_top1_ft = format_hm(all_avg_top1_ft, "red")
    all_avg_top1_ap = format_hm(all_avg_top1_ap, "blue")

    return (
        all_top1_lp,
        all_top1_ft,
        all_top1_ap,
        all_avg_top1_lp,
        all_avg_top1_ft,
        all_avg_top1_ap,
        res_results,
    )


# === BirdSet ===
def birdset_table(models, model_names, path, path_beans, finetuning, restricted, auroc):
    df = pd.read_csv(path, sep=",")

    # Rename for convenience
    df = df.rename(
        columns={
            "datamodule.dataset.dataset_name": "Dataset",
            "module.network.model_name": "Model",
            "tags": "Tags",
            "module.network.model.pooling": "Pooling",
            "test/cmAP5": "Cmap",
            "test/MultilabelAUROC": "auroc",
            "module.network.model.restrict_logits": "Restrict",
        }
    )

    # if the "restrict" is present in the tags, set Restrict to True
    # df["Restrict"] = df["Tags"].str.contains("restrict", case=False)

    # Convert scores to percentage
    metric = "Cmap" if not auroc else "auroc"

    df[metric] *= 100

    datasets = [
        "POW",
        "PER",
        "NES",
        "UHH",
        "HSN",
        "NBP",
        "SSW",
        "SNE",
    ]  # This will not change often

    # Delete old LaTeX file if it exists
    output_path = "projects/biofoundation/results/latex/results.tex"
    if os.path.exists(output_path):
        os.remove(output_path)

    # === Top table part ===
    stretch = 1 if finetuning else 1.5
    with open(output_path, "a") as f:
        f.write(
            "\\renewcommand{\\arraystretch}{"
            + str(stretch)
            + "} % Increase row height\n"
            "\\setlength{\\tabcolsep}{2pt}\n\n"
            "\\begin{tabular}{>{\\centering\\arraybackslash}p{0.7cm} p{1.5cm} | ccccc | >{\centering\\arraybackslash}p{0.8cm} !{\\vrule width 1.3pt} cccccccc | >{\centering\\arraybackslash}p{0.8cm}}\n"
            "    \\toprule\n"
            "    \\multicolumn{2}{c}{} & \\multicolumn{6}{c}{\\textbf{BEANS}} & \\multicolumn{9}{c}{\\makecell[c]{\\textbf{BirdSet} \\\\[-12pt] \\hspace{-7.5cm} {\\color{gray}\\scriptsize VAL}}}                                                                                                                                                                                                                                                                                                                                                                                  \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-17} % Ensuring cline matches actual columns\n"
            "    \\addlinespace[2pt]\n\n"
            "    \\multicolumn{2}{c}{} & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}}   & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HUM}} & \\cellcolor{gray!25}\\textbf{\\underline{Score}}"
            "                         & \\cellcolor{gray!25}\\textbf{\\textsc{POW}}   & \\cellcolor{gray!25}\\textbf{\\textsc{PER}} & \\cellcolor{gray!25}\\textbf{\\textsc{NES}} & \\cellcolor{gray!25}\\textbf{\\textsc{UHH}} & \\cellcolor{gray!25}\\textbf{\\textsc{HSN}} & \\cellcolor{gray!25}\\textbf{\\textsc{NBP}}   & \\cellcolor{gray!25}\\textbf{\\textsc{SSW}} & \\cellcolor{gray!25}\\textbf{\\textsc{SNE}} & \\cellcolor{gray!25}\\textbf{\\underline{Score}}                                                                         \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-17} % Adjusting cline to match new column numbers\n"
            "    \\addlinespace[2pt]\n"
        )

    # === Build table rows ===
    # Initialize lists to store all values for later processing
    all_cmap_lp, all_cmap_ft, all_cmap_ap = [], [], []
    all_avg_cmap_lp, all_avg_cmap_ft, all_avg_cmap_ap = [], [], []

    print(f"Processing BirdSet data for {len(models)} models...")

    # Collect data for all models
    for model in models:
        cmap_lp, cmap_ft, cmap_ap = [], [], []
        for dataset in datasets:
            lp_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("linearprobing"))
                & (df["Pooling"] != "attentive")
                & (df["Pooling"] != "average")
                & (df["Restrict"] != True)
            ]
            ft_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("finetune|finetuning"))
                & (df["Pooling"] != "attentive")
                & (df["Pooling"] != "average")
                & (df["Restrict"] != True)
            ]

            ap_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Pooling"] == "attentive")
                & (df["Restrict"] != True)
            ]

            cmap_lp.append(
                calculate_mean_std(lp_rows[metric].tolist()) if not lp_rows.empty else "-"
            )  # Mean±std for LP Cmap
            cmap_ft.append(
                calculate_mean_std(ft_rows[metric].tolist()) if not ft_rows.empty else "-"
            )  # Mean±std for FT Cmap
            cmap_ap.append(
                calculate_mean_std(ap_rows[metric].tolist()) if not ap_rows.empty else "-"
            )  # Mean±std for AP Cmap
        # Averages without the first (0th) column POW - calculate from the mean values
        # Extract numeric means from the tuples for average calculation
        lp_means = []
        ft_means = []
        ap_means = []
        
        for i, val in enumerate(cmap_lp):
            if i != 0 and val != "-" and isinstance(val, tuple) and len(val) == 2 and val[0] is not None:  # Skip POW (index 0)
                lp_means.append(val[0])  # Extract mean from tuple
        for i, val in enumerate(cmap_ft):
            if i != 0 and val != "-" and isinstance(val, tuple) and len(val) == 2 and val[0] is not None:  # Skip POW (index 0)
                ft_means.append(val[0])  # Extract mean from tuple
        for i, val in enumerate(cmap_ap):
            if i != 0 and val != "-" and isinstance(val, tuple) and len(val) == 2 and val[0] is not None:  # Skip POW (index 0)
                ap_means.append(val[0])  # Extract mean from tuple
        
        avg_cmap_lp = calculate_mean_std(lp_means)
        avg_cmap_ft = calculate_mean_std(ft_means)
        avg_cmap_ap = calculate_mean_std(ap_means)

        # Store all values for later processing
        all_cmap_lp.append(cmap_lp)
        all_cmap_ft.append(cmap_ft)
        all_cmap_ap.append(cmap_ap)
        all_avg_cmap_ap.append(avg_cmap_ap)
        all_avg_cmap_lp.append(avg_cmap_lp)
        all_avg_cmap_ft.append(avg_cmap_ft)

    print(f"BirdSet data collected for {len(all_cmap_lp)} models")

    # Convert tuples to strings and format
    processed_cmap_lp = []
    for model_data in all_cmap_lp:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_cmap_lp.append(model_row)
    all_cmap_lp = format_values(processed_cmap_lp)

    processed_cmap_ft = []
    for model_data in all_cmap_ft:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_cmap_ft.append(model_row)
    all_cmap_ft = format_values(processed_cmap_ft)
    
    processed_cmap_ap = []
    for model_data in all_cmap_ap:
        model_row = []
        for val in model_data:
            if isinstance(val, tuple) and len(val) == 2:
                mean_val, std_val = val
                if mean_val is not None:
                    model_row.append(format_mean_std_display(mean_val, std_val))
                else:
                    model_row.append("-")
            else:
                model_row.append("-")
        processed_cmap_ap.append(model_row)
    all_cmap_ap = format_values(processed_cmap_ap)

    # Format cmap values for heatmap
    all_avg_cmap_lp = format_hm(all_avg_cmap_lp, "green")
    all_avg_cmap_ft = format_hm(all_avg_cmap_ft, "red")
    all_avg_cmap_ap = format_hm(all_avg_cmap_ap, "blue")

    # Get BEANS results and unpack lists, adding _beans to the end of each variable name

    (
        all_top1_lp_beans,
        all_top1_ft_beans,
        all_top1_ap_beans,
        all_avg_top1_lp_beans,
        all_avg_top1_ft_beans,
        all_avg_top1_ap_beans,
        res_results_beans,
    ) = beans_table(path_beans, models, restricted, auroc)

    with open(output_path, "a") as f:
        for i, model in enumerate(models):

            # Write LaTeX to a file
            num_rows = 3 if finetuning else 2
            f.write(
                f"\\multirow{{{num_rows}}}{{*}}{{\\textbf{{{format_name(model_names[i])}}}}} & {{Linear}} & "
                + " & ".join(all_top1_lp_beans[i])
                + f" & {all_avg_top1_lp_beans[i]} &"
                + " & ".join(all_cmap_lp[i])
                + f" & {all_avg_cmap_lp[i]} \\\\ \n"
            )
            if restricted and (
                model == "surfperch" or model == "perch" or model == "convnext_bs"
            ):
                # Calculate restricted results in isolated form for easy removal
                cmap_res = []
                for dataset in datasets:
                    res_rows = df[
                        (df["Model"] == model)
                        & (df["Dataset"] == dataset)
                        & (df["Restrict"] == True)
                    ]
                    res_values = res_rows[metric].tolist() if not res_rows.empty else []
                    cmap_res.append(calculate_mean_std(res_values))
                
                # Calculate average excluding POW (index 0)
                res_means = []
                for i, val in enumerate(cmap_res):
                    if i != 0 and val != "-" and "$\\pm$" in val:  # Skip POW (index 0)
                        res_means.append(float(val.split("($\\pm$")[0]))
                
                avg_cmap_res = calculate_mean_std(res_means)
                
                # Do the formatting separately
                cmap_res = format_values_no_bold(cmap_res)
                avg_cmap_res = format_hm_no_bold([avg_cmap_res], "blue")[0]

                # Add the restricted results to the beans cbi results
                cbi_results = ["-"] * len(all_top1_ap_beans[i])
                cbi_results[2] = res_results_beans[model]["top1"]
                # TODO: Add for beans if it works with CBI
                f.write(
                    f" & {{Restricted}} & "
                    + " & ".join(cbi_results)
                    + f" & - &"
                    + " & ".join(cmap_res)
                    + f" & {avg_cmap_res} \\\\ \n"
                )
            else:
                f.write(
                    f" & {{Attentive}} & "
                    + " & ".join(all_top1_ap_beans[i])
                    + f" & {all_avg_top1_ap_beans[i]} &"
                    + " & ".join(all_cmap_ap[i])
                    + f" & {all_avg_cmap_ap[i]} \\\\ \n"
                )

            if finetuning:
                f.write(
                    f" & {{Finetuned}} & "
                    + " & ".join(all_top1_ft_beans[i])
                    + f" & {all_avg_top1_ft_beans[i]} &"
                    + " & ".join(all_cmap_ft[i])
                    + f" & {all_avg_cmap_ft[i]} \\\\ \n"
                )

            if model != models[-1]:
                f.write(f"\\hline \n")

    # === Table end part ===
    with open(output_path, "a") as f:
        f.write("    \\bottomrule\n")
        f.write("\\end{tabular}\n")


# === Script ===
print(f"Creating Latex table...")
# Settings:
MODELS = [
    "audio_mae",
    "aves",
    "BEATs",
    "BEATs_NatureLM",
    "biolingual",
    "bird_aves",
    "birdmae",
    "convnext_bs",
    "eat_ssl",
    "perch",
    "ProtoCLR",
    "surfperch",
    "vit_inatsound",
]  # Extract these names from the CSV file
MODEL_NAMES = [
    "Audio-MAE",
    "AVES",
    "BEATs",
    "BEATs-NLM",
    "Biolin-gual",
    "Bird-AVES",
    "Bird-MAE",
    "Conv-Next$_{BS}$",
    "EAT-SSL",
    "Perch",
    "Proto-CLR",
    "Surf-Perch",
    "ViT-INS",
]  # These names will appear in the table split at "-" ordered the same as MODELS
CSV_PATH_BEANS = "projects/biofoundation/results/latex/beans.csv"
CSV_PATH = "projects/biofoundation/results/latex/birdset.csv"
FINETUNING = False  # Set to True if you want to include finetuning results
RESTRICTED = True  # Set to True to use Perch, Surfperch, Convnext_Bs restricted models
AUROC = True  # Set to True to use AUROC instead of Top1

# Print summary of settings
print("Summary of settings:")
print(f"Models: {len(MODELS)}")
for model, model_name in zip(MODELS, MODEL_NAMES):
    print(f"  {model} -> {model_name}")
print(f"Finetuning: {FINETUNING}")

birdset_table(
    MODELS, MODEL_NAMES, CSV_PATH, CSV_PATH_BEANS, FINETUNING, RESTRICTED, AUROC
)
