import pandas as pd
import numpy as np
import os
import sys

import numpy
# === Functions ===
# Format row with LaTeX
def format_values(values):
    # Transpose the list of lists to treat each position as a column
    columns = np.array(values).T
    formatted_columns = []

    for col in columns:
        rounded = np.round(col, 1)
        max_idx = np.argmax(rounded)
        second_max_idx = np.argsort(rounded)[-2]
        formatted = [f"{val:.1f}" if val > 0 else "-" for val in rounded]
        formatted[max_idx] = f"\\textbf{{{rounded[max_idx]:.1f}}}"
        formatted[second_max_idx] = f"\\underline{{{rounded[second_max_idx]:.1f}}}"
        formatted_columns.append(formatted)

    # Transpose back to match the original structure
    return np.array(formatted_columns).T.tolist()


def format_name(name):
    if "-" in name:
        parts = name.split("-")
        name = (
            "\\rotatebox[origin=c]{90}{\\begin{tabular}{@{}c@{}}"
            + " \\\\ ".join(parts)
            + "\\end{tabular}}\n"
        )
    else:
        name = "\\rotatebox[origin=c]{90}{" + name + "}\n"
    return name


def format_hm(values, color):
    rounded = np.round(values, 1)
    max_idx = np.argmax(rounded)
    second_max_idx = np.argsort(rounded)[-2]
    formatted = [f"\\heat{color}{{{val:.1f}}}" if val > 0 else "-" for val in rounded]
    formatted[max_idx] = f"\\heat{color}[bold]{{{rounded[max_idx]:.1f}}}"
    formatted[second_max_idx] = (
        f"\\heat{color}[underline]{{{rounded[second_max_idx]:.1f}}}"
    )
    return formatted


# === BEANS === (Adjusted for one table so it returns the needed lists)
def beans_table(path, models):
    df = pd.read_csv(path, sep=",")

    # Rename for convenience
    df = df.rename(
        columns={
            "datamodule.dataset.dataset_name": "Dataset",
            "module.network.model_name": "Model",
            "tags": "Tags",
            "module.network.model.pooling": "Pooling",
            "test/MulticlassAccuracy": "Top1",
        }
    )

    # Convert scores to percentage
    df["Top1"] *= 100

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

    # Collect data for all models
    for model in models:
        top1_lp, top1_ft, top1_ap = [], [], []

        for dataset in datasets:
            lp_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("linearprobing"))
            ]
            ft_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("finetune|finetuning"))
                & (df["Pooling"] != "attentive")
            ]

            ap_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Pooling"] == "attentive")
            ]

            top1_lp.append(
                lp_rows["Top1"].max() if not lp_rows.empty else 0
            )  # Max value for LP Top1
            top1_ft.append(
                ft_rows["Top1"].max() if not ft_rows.empty else 0
            )  # Max value for FT Top1
            top1_ap.append(
                ap_rows["Top1"].max() if not ap_rows.empty else 0
            )  # Max value for AP Top1
        # Averages
        avg_top1_lp = (
            np.round(np.mean([x for x in top1_lp if x > 0]), 1)
            if any(x > 0 for x in top1_lp)
            else 0
        )
        avg_top1_ft = (
            np.round(np.mean([x for x in top1_ft if x > 0]), 1)
            if any(x > 0 for x in top1_ft)
            else 0
        )
        avg_top1_ap = (
            np.round(np.mean([x for x in top1_ap if x > 0]), 1)
            if any(x > 0 for x in top1_ap)
            else 0
        )

        # Store all values for later processing
        all_top1_lp.append(top1_lp)
        all_top1_ft.append(top1_ft)
        all_top1_ap.append(top1_ap)
        all_avg_top1_lp.append(avg_top1_lp)
        all_avg_top1_ft.append(avg_top1_ft)
        all_avg_top1_ap.append(avg_top1_ap)

    # Determine the highest and second highest values and write LaTeX
    all_top1_lp = format_values(all_top1_lp)
    all_top1_ft = format_values(all_top1_ft)
    all_top1_ap = format_values(all_top1_ap)

    all_avg_top1_lp = format_hm(all_avg_top1_lp, "green")
    all_avg_top1_ft = format_hm(all_avg_top1_ft, "red")
    all_avg_top1_ap = format_hm(all_avg_top1_ap, "blue")

    return all_top1_lp, all_top1_ft, all_top1_ap, all_avg_top1_lp, all_avg_top1_ft, all_avg_top1_ap



# === BirdSet ===
def birdset_table(models, model_names, path, path_beans):
    df = pd.read_csv(path, sep=",")

    # Rename for convenience
    df = df.rename(
        columns={
            "datamodule.dataset.dataset_name": "Dataset",
            "module.network.model_name": "Model",
            "tags": "Tags",
            "module.network.model.pooling": "Pooling",
            "test/cmAP5": "Cmap",
        }
    )

    # Convert scores to percentage
    df["Cmap"] *= 100

    datasets = [
        "PER",
        "POW",
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
    with open(output_path, "a") as f:
        f.write(
            "\\renewcommand{\\arraystretch}{1} % Increase row height\n"
            "\\setlength{\\tabcolsep}{2pt}\n\n"
            "\\begin{tabular}{p{0.7cm} p{1.5cm} | ccccc | >{\centering\\arraybackslash}p{0.8cm} !{\\vrule width 1.3pt} cccccccc | >{\centering\\arraybackslash}p{0.8cm}}\n"
            "    \\toprule\n"
            "    \\multicolumn{2}{c}{} & \\multicolumn{6}{c}{\\textbf{BEANS}} & \\multicolumn{9}{c}{\\makecell[c]{\\textbf{BirdSet} \\\\[-12pt] \\hspace{-5.6cm} {\\color{gray}\\scriptsize VAL}}}                                                                                                                                                                                                                                                                                                                                                                                  \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-17} % Ensuring cline matches actual columns\n"
            "    \\addlinespace[2pt]\n\n"
            "    \\multicolumn{2}{c}{} & \\cellcolor{gray!25}\\textbf{\\textsc{WTK}}   & \\cellcolor{gray!25}\\textbf{\\textsc{BAT}} & \\cellcolor{gray!25}\\textbf{\\textsc{CBI}} & \\cellcolor{gray!25}\\textbf{\\textsc{DOG}} & \\cellcolor{gray!25}\\textbf{\\textsc{HUM}} & \\cellcolor{gray!25}\\textbf{\\underline{Score}}"
            "                         & \\cellcolor{gray!25}\\textbf{\\textsc{PER}}   & \\cellcolor{gray!25}\\textbf{\\textsc{POW}} & \\cellcolor{gray!25}\\textbf{\\textsc{NES}} & \\cellcolor{gray!25}\\textbf{\\textsc{UHH}} & \\cellcolor{gray!25}\\textbf{\\textsc{HSN}} & \\cellcolor{gray!25}\\textbf{\\textsc{NBP}}   & \\cellcolor{gray!25}\\textbf{\\textsc{SSW}} & \\cellcolor{gray!25}\\textbf{\\textsc{SNE}} & \\cellcolor{gray!25}\\textbf{\\underline{Score}}                                                                         \\\\\n"
            "    \\addlinespace[2pt]\n"
            "    \\cline{3-17} % Adjusting cline to match new column numbers\n"
            "    \\addlinespace[2pt]\n"
        )

    # === Build table rows ===
    # Initialize lists to store all values for later processing
    all_cmap_lp, all_cmap_ft, all_cmap_ap = [], [], []
    all_avg_cmap_lp, all_avg_cmap_ft, all_avg_cmap_ap = [], [], []

    # Collect data for all models
    for model in models:
        cmap_lp, cmap_ft, cmap_ap = [], [], []

        for dataset in datasets:
            lp_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("linearprobing"))
            ]
            ft_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Tags"].str.contains("finetune|finetuning"))
                & (df["Pooling"] != "attentive")
            ]

            ap_rows = df[
                (df["Model"] == model)
                & (df["Dataset"] == dataset)
                & (df["Pooling"] == "attentive")
            ]

            cmap_lp.append(
                lp_rows["Cmap"].max() if not lp_rows.empty else 0
            )  # Max value for LP Cmap
            cmap_ft.append(
                ft_rows["Cmap"].max() if not ft_rows.empty else 0
            )  # Max value for FT Cmap
            cmap_ap.append(
                ap_rows["Cmap"].max() if not ap_rows.empty else 0
            )
        # Averages without the second (1) column POW
        avg_cmap_lp = (
            round(np.mean([x for i, x in enumerate(cmap_lp) if x > 0 and i != 1]), 1)
            if any(x > 0 and i != 1 for i, x in enumerate(cmap_lp))
            else 0
        )
        avg_cmap_ft = (
            round(np.mean([x for i, x in enumerate(cmap_ft) if x > 0 and i != 1]), 1)
            if any(x > 0 and i != 1 for i, x in enumerate(cmap_ft))
            else 0
        )

        avg_cmap_ap = (
            round(np.mean([x for i, x in enumerate(cmap_ap) if x > 0 and i != 1]), 1)
            if any(x > 0 and i != 1 for i, x in enumerate(cmap_ap))
            else 0
        )

        # Store all values for later processing
        all_cmap_lp.append(cmap_lp)
        all_cmap_ft.append(cmap_ft)
        all_cmap_ap.append(cmap_ap)
        all_avg_cmap_ap.append(avg_cmap_ap)
        all_avg_cmap_lp.append(avg_cmap_lp)
        all_avg_cmap_ft.append(avg_cmap_ft)

    # Determine the highest and second highest values and write LaTeX
    all_cmap_lp = format_values(all_cmap_lp)
    all_cmap_ft = format_values(all_cmap_ft)
    all_cmap_ap = format_values(all_cmap_ap)

    # Format cmap values for heatmap
    all_avg_cmap_lp = format_hm(all_avg_cmap_lp, "green")
    all_avg_cmap_ft = format_hm(all_avg_cmap_ft, "red")
    all_avg_cmap_ap = format_hm(all_avg_cmap_ap, "blue")

    # Get BEANS results and unpack lists, adding _beans to the end of each variable name
    
    all_top1_lp_beans, all_top1_ft_beans, all_top1_ap_beans, all_avg_top1_lp_beans, all_avg_top1_ft_beans, all_avg_top1_ap_beans = beans_table(path_beans, models)

    with open(output_path, "a") as f:
        for i, model in enumerate(models):

            # Write LaTeX to a file
            f.write(
                f"\\multirow{{3}}{{*}}{{\\textbf{{{format_name(model_names[i])}}}}} & {{Linear}} & "
                + " & ".join(all_top1_lp_beans[i]) + f" & {all_avg_top1_lp_beans[i]} &"
                + " & ".join(all_cmap_lp[i]) + f" & {all_avg_cmap_lp[i]} \\\\ \n"
            )

            f.write(
                f" & {{Attentive}} & "
                + " & ".join(all_top1_ap_beans[i]) + f" & {all_avg_top1_ap_beans[i]} &"
                + " & ".join(all_cmap_ap[i]) + f" & {all_avg_cmap_ap[i]} \\\\ \n"
            )

            f.write(
                f" & {{Finetuned}} & "
                + " & ".join(all_top1_ft_beans[i]) + f" & {all_avg_top1_ft_beans[i]} &"
                + " & ".join(all_cmap_ft[i]) + f" & {all_avg_cmap_ft[i]} \\\\ \n"
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
MODELS = ["audio_mae", "aves", "BEATs", "biolingual", "bird_aves", "bird_mae", "convnext_bs", "eat_ssl", "BEATs_NatureLM", "perch", "proto_clr", "surfperch", "vit_ins"] # Extract these names from the CSV file
MODEL_NAMES = ["Audio-MAE", "AVES", "BEATs", "Bioli-ngual", "Bird-AVES", "Bird-MAE", "Conv-NextBS", "EAT-SSL", "BEATs-NLM", "Perch", "Proto-CLR", "Surf-Perch", "ViT-INS"] # These names will appear in the table split at "-" ordered the same as MODELS
CSV_PATH_BEANS = "projects/biofoundation/results/latex/beans.csv"
CSV_PATH = "projects/biofoundation/results/latex/birdset.csv"

birdset_table(MODELS, MODEL_NAMES, CSV_PATH, CSV_PATH_BEANS)