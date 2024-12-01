import pandas as pd, numpy as np
import argparse
import pickle
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.utils import resample
from collections import defaultdict

def pickle_to_res(path):
    with open(path, "rb") as f:
        raw_pickle = pickle.load(f)

    results = {}
    answer_prefix="Answer Choice "

    def extract_ans(ans_str, search_substring):
        # Split the string into lines
        lines = ans_str.split("\n")

        for line in lines:
            # Check if the line starts with the specified substring
            if line.startswith(search_substring):
                # If it does, add it to the list of extracted rows
                return line[len(search_substring) :].strip()
        return "ERROR"  # Answer not found

    for k, v in raw_pickle.items():
        if k != 'token_usage':  # Skip token_usage
            qns_idx = ast.literal_eval(k)
            for idx, qn_idx in enumerate(qns_idx):
                results[qn_idx] = extract_ans(
                    v[0], f"{answer_prefix}{idx+1}:"
                )  # We start with question 1
    
    return results


def res_to_vec(res):
    test_df = pd.read_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_test_metadata.csv', index_col=0)
    num_errors = 0
    labels, preds, race = [], [], []
    for i in test_df.itertuples():
        fst = '12' if i.skin_tone == 12 else '56'
        if (i.Index not in res) or (res[i.Index].startswith("ERROR")):
            num_errors += 1

            print(i.Index, f"answer not found")
            continue

        pred_text = res[i.Index]
        ground_truth = "B" if i.malignant == True else "A"
        labels.append(ground_truth)
        preds.append(pred_text)
        race.append(fst)
        
    print(f"In total {num_errors} errors len = {len(labels)}")
    return labels,preds,race

def calculate_metrics(actual, predicted):
    TP = sum((a == 'B' and p == 'B') for a, p in zip(actual, predicted))
    TN = sum((a == 'A' and p == 'A') for a, p in zip(actual, predicted))
    FP = sum((a == 'A' and p == 'B') for a, p in zip(actual, predicted))
    FN = sum((a == 'B' and p == 'A') for a, p in zip(actual, predicted))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    accuracy = (TP + TN) / len(actual)
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return accuracy, tpr, tnr, f1_score



def filter_by_race(actual, predicted, race, race_value):
    filtered_actual = [a for a, r in zip(actual, race) if r == race_value]
    filtered_predicted = [p for p, r in zip(predicted, race) if r == race_value]
    return filtered_actual, filtered_predicted

def bootstrap_metrics(labels, preds, race=None, race_filter=None, n_iterations=1000):
    """
    Calculate bootstrapped metrics with confidence intervals
    """
    if race is not None and race_filter is not None:
        # Filter for specific race group
        mask = np.array(race) == race_filter
        labels = np.array(labels)[mask]
        preds = np.array(preds)[mask]
    
    n_samples = len(labels)
    accuracies = []
    tprs = []
    fscores = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        idx = resample(range(n_samples),replace=True)
        boot_labels = [labels[i] for i in idx]
        boot_preds = [preds[i] for i in idx]
        
        accuracy, tpr, _, fscore = calculate_metrics(boot_labels, boot_preds)
        accuracies.append(accuracy)
        tprs.append(tpr)
        fscores.append(fscore)
    
    # Calculate mean and standard deviation
    metrics_mean = {
        'accuracy': np.mean(accuracies),
        'tpr': np.mean(tprs),
        'fscore': np.mean(fscores)
    }
    
    metrics_std = {
        'accuracy': np.std(accuracies),
        'tpr': np.std(tprs),
        'fscore': np.std(fscores)
    }
    
    return metrics_mean, metrics_std

def calculate_bootstrap_metrics(exps):
    """
    Calculate bootstrap metrics for all experiments
    """
    bulk_metrics = {'mean': [], 'std': []}
    fst12_metrics = {'mean': [], 'std': []}
    fst56_metrics = {'mean': [], 'std': []}
    
    for exp in exps:
        results = pickle_to_res(exp)
        labels, preds, race = res_to_vec(results)
        
        # Calculate bulk metrics
        bulk_mean, bulk_std = bootstrap_metrics(labels, preds)
        bulk_metrics['mean'].append(bulk_mean)
        bulk_metrics['std'].append(bulk_std)
        
        # Calculate FST 1/2 metrics
        fst12_mean, fst12_std = bootstrap_metrics(labels, preds, race, '12')
        fst12_metrics['mean'].append(fst12_mean)
        fst12_metrics['std'].append(fst12_std)
        
        # Calculate FST 5/6 metrics
        fst56_mean, fst56_std = bootstrap_metrics(labels, preds, race, '56')
        fst56_metrics['mean'].append(fst56_mean)
        fst56_metrics['std'].append(fst56_std)
    
    return bulk_metrics, fst12_metrics, fst56_metrics

def filter_by_race(actual, predicted, race, race_value):
    filtered_actual = [a for a, r in zip(actual, race) if r == race_value]
    filtered_predicted = [p for p, r in zip(predicted, race) if r == race_value]
    return filtered_actual, filtered_predicted

import numpy as np
import matplotlib.pyplot as plt

def plot_experiment_results(bulk_metrics, fst12_metrics, fst56_metrics, score_type, shots, use_error_bars=True, label=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use custom colors for the lines
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Calculate statistics for each experiment
    bulk_means = [np.mean(exp) for exp in bulk_metrics['mean']]
    bulk_stds = [np.std(exp) for exp in bulk_metrics['std']]
    bulk_mins = [np.min(exp) for exp in bulk_metrics['mean']]
    bulk_maxs = [np.max(exp) for exp in bulk_metrics['mean']]
    bulk_ranges = [bulk_maxs[i] - bulk_mins[i] for i in range(len(bulk_mins))]

    fst12_means = [np.mean(exp) for exp in fst12_metrics['mean']]
    fst12_stds = [np.std(exp) for exp in fst12_metrics['std']]
    fst12_mins = [np.min(exp) for exp in fst12_metrics['mean']]
    fst12_maxs = [np.max(exp) for exp in fst12_metrics['mean']]
    fst12_ranges = [fst12_maxs[i] - fst12_mins[i] for i in range(len(fst12_mins))]

    fst56_means = [np.mean(exp) for exp in fst56_metrics['mean']]
    fst56_stds = [np.std(exp) for exp in fst56_metrics['std']]
    fst56_mins = [np.min(exp) for exp in fst56_metrics['mean']]
    fst56_maxs = [np.max(exp) for exp in fst56_metrics['mean']]
    fst56_ranges = [fst56_maxs[i] - fst56_mins[i] for i in range(len(fst56_mins))]

    
    ax1.errorbar(shots, bulk_means, yerr=bulk_ranges, fmt='o-', 
                color=line_colors[0], label=f'Aggregate {score_type}', 
                linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
    ax1.errorbar(shots, fst12_means, yerr=fst12_ranges, fmt='s-',
                color=line_colors[1], label=f'FST 1/2 {score_type}',
                linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
    ax1.errorbar(shots, fst56_means, yerr=fst56_ranges, fmt='^-',
                color=line_colors[2], label=f'FST 5/6 {score_type}',
                linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
    ax1.set_xlabel('Number of Shots')
    ax1.set_ylabel(f'{score_type}')
    ax1.set_title(f'{score_type} by Number of Shots and Skin Type')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot bias with confidence intervals
    biases_mean = np.array(fst12_means) - np.array(fst56_means)
    bias_mins = np.array(fst12_mins) - np.array(fst56_maxs)
    bias_maxs = np.array(fst12_maxs) - np.array(fst56_mins)
    bias_ranges = bias_maxs - bias_mins

    ax2.errorbar(shots, biases_mean, yerr=bias_ranges, fmt='o-',
                color='purple', label=f'Bias (FST 1/2 - FST 5/6 {score_type})',
                linewidth=2, capsize=5, capthick=1, elinewidth=1)

    ax2.set_xlabel('Number of Shots')
    ax2.set_ylabel('Bias')
    ax2.set_title('Bias by Number of Shots')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig_name = f"{label}_{score_type}_by_shots_and_skin_type"
    fig.savefig("Multiple_experiments_plots/" + fig_name + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_experiment_metrics_with_ci(bulk_metrics, fst12_metrics, fst56_metrics, score_type, shots, use_error_bars=False, label=""):
    """
    Plot experiment metrics with confidence intervals
    
    Parameters:
    bulk_metrics, fst12_metrics, fst56_metrics: dictionaries containing means and standard deviations
    score_type: string indicating the type of score being plotted ('accuracy', 'tpr', or 'fscore')
    shots: list of integers representing the number of shots for each experiment
    use_error_bars: boolean indicating whether to use error bars instead of fill-between for uncertainty
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use custom colors for the lines
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Extract means and standard deviations
    bulk_means = [m[score_type.lower()] for m in bulk_metrics['mean']]
    bulk_stds = [s[score_type.lower()] for s in bulk_metrics['std']]
    
    fst12_means = [m[score_type.lower()] for m in fst12_metrics['mean']]
    fst12_stds = [s[score_type.lower()] for s in fst12_metrics['std']]
    
    fst56_means = [m[score_type.lower()] for m in fst56_metrics['mean']]
    fst56_stds = [s[score_type.lower()] for s in fst56_metrics['std']]
    
    # Plot lines and confidence intervals
    if use_error_bars:
        ax1.errorbar(shots, bulk_means, yerr=bulk_stds, fmt='o-', 
                    color=line_colors[0], label=f'Aggregate {score_type}', 
                    linewidth=2, capsize=5, capthick=1, elinewidth=1)
        
        ax1.errorbar(shots, fst12_means, yerr=fst12_stds, fmt='s-',
                    color=line_colors[1], label=f'FST 1/2 {score_type}',
                    linewidth=2, capsize=5, capthick=1, elinewidth=1)
        
        ax1.errorbar(shots, fst56_means, yerr=fst56_stds, fmt='^-',
                    color=line_colors[2], label=f'FST 5/6 {score_type}',
                    linewidth=2, capsize=5, capthick=1, elinewidth=1)
    else:
        ax1.plot(shots, bulk_means, 'o-', color=line_colors[0], label=f'Aggregate {score_type}', linewidth=2)
        ax1.fill_between(shots, 
                        np.array(bulk_means) - np.array(bulk_stds),
                        np.array(bulk_means) + np.array(bulk_stds),
                        color=line_colors[0], alpha=0.2)
        
        ax1.plot(shots, fst12_means, 's-', color=line_colors[1], label=f'FST 1/2 {score_type}', linewidth=2)
        ax1.fill_between(shots,
                        np.array(fst12_means) - np.array(fst12_stds),
                        np.array(fst12_means) + np.array(fst12_stds),
                        color=line_colors[1], alpha=0.2)
        
        ax1.plot(shots, fst56_means, '^-', color=line_colors[2], label=f'FST 5/6 {score_type}', linewidth=2)
        ax1.fill_between(shots,
                        np.array(fst56_means) - np.array(fst56_stds),
                        np.array(fst56_means) + np.array(fst56_stds),
                        color=line_colors[2], alpha=0.2)
    
    ax1.set_xlabel('Number of Shots')
    ax1.set_ylabel(f'{score_type}')
    ax1.set_title(f'{score_type} by Number of Shots and Skin Type ({label})')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot bias with confidence intervals
    biases_mean = np.array(fst12_means) - np.array(fst56_means)
    biases_std = np.sqrt(np.array(fst12_stds)**2 + np.array(fst56_stds)**2)  # Error propagation
    
    if use_error_bars:
        ax2.errorbar(shots, biases_mean, yerr=biases_std, fmt='o-',
                    color='purple', label=f'Bias (FST 1/2 - FST 5/6 {score_type})',
                    linewidth=2, capsize=5, capthick=1, elinewidth=1)
    else:
        ax2.plot(shots, biases_mean, 'o-', color='purple', 
                label=f'Bias (FST 1/2 - FST 5/6 {score_type})', linewidth=2)
        ax2.fill_between(shots,
                        biases_mean - biases_std,
                        biases_mean + biases_std,
                        color='purple', alpha=0.2)
    
    ax2.set_xlabel('Number of Shots')
    ax2.set_ylabel('Bias')
    ax2.set_title(f'Bias by Number of Shots ({label})')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig_name = f"{label}_{score_type}_by_shots_and_skin_type"
    fig.savefig(fig_name + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def parse_experiment_name(filename, model_name="gpt", num_shots=50):
    """Parses the experiment name to extract relevant information."""
    normalized_filename = filename.replace("\\", "/")
    pattern = (
        rf"ddi_results/ddi_(?P<fst12_ben>\d+)_(?P<fst12_mal>\d+)_"
        rf"(?P<fst56_ben>\d+)_(?P<fst56_mal>\d+)_{model_name}_" 
        rf"{num_shots}\.pkl"
    )
    match = re.match(pattern, normalized_filename) 
    if match:
        return {
            "fst12_ben": int(match.group("fst12_ben")),
            "fst12_mal": int(match.group("fst12_mal")),
            "fst56_ben": int(match.group("fst56_ben")),
            "fst56_mal": int(match.group("fst56_mal"))
        }
    else:
        return None

def sort_experiments(experiment_names, model_name="gpt", num_shots=50):
    """Sorts experiments based on the specified criteria."""

    experiments = {
        "matched_fst12_fst56": [],
        "matched_balanced_fst12_fst56": [],
        "fst12_only": [],
        "fst56_only": [],
        "balanced_fst12_only": [],
        "balanced_fst56_only": []
    }

    for filename in experiment_names:
        experiment_info = parse_experiment_name(filename, model_name=model_name, num_shots=num_shots)
        print(experiment_info)
        if all(val == 0 for val in experiment_info.values()):
            for key in experiments.keys():
                experiments[key].append(filename)
            continue
        if experiment_info:
            if experiment_info["fst12_ben"] == experiment_info["fst56_ben"] and experiment_info["fst12_mal"] == experiment_info["fst56_mal"] and experiment_info["fst12_ben"] == experiment_info["fst12_mal"]:
                experiments["matched_balanced_fst12_fst56"].append(filename)
            elif experiment_info["fst12_ben"] == experiment_info["fst56_ben"] and experiment_info["fst12_mal"] == experiment_info["fst56_mal"]:
                experiments["matched_fst12_fst56"].append(filename)
            elif experiment_info["fst56_ben"] == 0 and experiment_info["fst56_mal"] == 0:
                if experiment_info["fst12_ben"] == experiment_info["fst12_mal"]:
                    experiments["balanced_fst12_only"].append(filename)
                else:
                    experiments["fst12_only"].append(filename)
            elif experiment_info["fst12_ben"] == 0 and experiment_info["fst12_mal"] == 0:
                if experiment_info["fst56_ben"] == experiment_info["fst56_mal"]:
                    experiments["balanced_fst56_only"].append(filename)
                else:
                    experiments["fst56_only"].append(filename)

    # Sort experiments within each category
    for category, filenames in experiments.items():
        def sort_key(filename):
            experiment_info = parse_experiment_name(filename, model_name=model_name, num_shots=num_shots)
            if experiment_info["fst12_ben"] == 0:
                return experiment_info["fst56_ben"]
            else:
                return experiment_info["fst12_ben"]
        
        filenames.sort(key=sort_key)
        experiments[category] = (filenames, sorted([sort_key(filename) for filename in filenames]))

    return experiments


# Main script
if __name__ == "__main__":
    models = ["gpt-4o-2024-05-13"]
    exps = Path('./ddi_results').glob('*.pkl')  # find all pickle files in results folder
    exps = [str(exp) for exp in exps]
    print(len(exps), exps)
    # Loop through each experiment
    for model in models:
        model_exps = [exp for exp in exps if model in exp]
        print(len(model_exps), model_exps)
        sorted_experiments = sort_experiments(model_exps, model_name=model)
        print(len(sorted_experiments), sorted_experiments)
        if len(sorted_experiments) == 0:
            print(f"No experiments found for model: {model}")
            continue
    
        for category, (filenames, shots) in sorted_experiments.items():
            print(shots)
            print(f"Processing category: {category}")
            if len(filenames) == 0:
                print(f"No experiments found for category: {category}")
                continue
            overall_bulk_metrics = []
            overall_fst12_metrics = []
            overall_fst56_metrics = []
            filenames_by_subdir = defaultdict(list)
            for filename in filenames:
                subdir = Path(filename).parent.name
                filenames_by_subdir[subdir].append(filename)
            print(filenames_by_subdir)
            for subdir, subdir_filenames in filenames_by_subdir.items():
                bulk_metrics, fst12_metrics, fst56_metrics = calculate_bootstrap_metrics(filenames)
                overall_bulk_metrics.append(bulk_metrics)
                overall_fst12_metrics.append(fst12_metrics)
                overall_fst56_metrics.append(fst56_metrics)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "accuracy", shots, use_error_bars=True, label=category) # overall_bulk_metrics, fst12_metrics, fst56_metrics, "accuracy", shots, use_error_bars=True, label=category)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "tpr", shots, use_error_bars=True, label=category)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "fscore", shots, use_error_bars=True, label=category)