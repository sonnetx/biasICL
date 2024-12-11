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
        if "90" in exp:
            continue
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

def extract_stats(metrics, score_type):
    """
    Extracts and calculates statistics (means, stds, min, max, ranges) for given metrics and score_type.
    """
    means = [np.array([d['mean'][i][score_type] for d in metrics]) for i in range(len(metrics[0]['mean']))]
    stds = [np.array([d['std'][i][score_type] for d in metrics]) for i in range(len(metrics[0]['std']))]

    means_aggregated = [np.mean(m) for m in means]
    stds_aggregated = [np.mean(s) for s in stds]  # Correcting earlier use.
    mins = [np.min(m) for m in means]
    maxs = [np.max(m) for m in means]
    ranges = [max_val - min_val for max_val, min_val in zip(maxs, mins)]

    return means_aggregated, stds_aggregated, mins, maxs, ranges

def plot_experiment_results(bulk_metrics, fst12_metrics, fst56_metrics, score_type, shots, use_error_bars=True, label=""):
    # Extract stats for the 3 sets
    bulk_means, bulk_stds, bulk_mins, bulk_maxs, bulk_ranges = extract_stats(bulk_metrics, score_type)
    fst12_means, fst12_stds, fst12_mins, fst12_maxs, fst12_ranges = extract_stats(fst12_metrics, score_type)
    fst56_means, fst56_stds, fst56_mins, fst56_maxs, fst56_ranges = extract_stats(fst56_metrics, score_type)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Error Bars (means & range for clear diff use)
    ax1.errorbar(shots, bulk_means, yerr=bulk_ranges if use_error_bars else None, fmt='o-', color=line_colors[0],
                 label=f'Aggregate {score_type}', linewidth=2, capsize=5)
    ax1.errorbar(shots, fst12_means, yerr=fst12_ranges if use_error_bars else None, fmt='s-', color=line_colors[1],
                 label=f'FST 1/2 {score_type}', linewidth=2, capsize=5)
    ax1.errorbar(shots, fst56_means, yerr=fst56_ranges if use_error_bars else None, fmt='^-', color=line_colors[2],
                 label=f'FST 5/6 {score_type}', linewidth=2, capsize=5)

    ax1.set_xlabel('Number of Shots')
    ax1.set_ylabel(f'{score_type}')
    ax1.set_title(f'{score_type} by Number of Shots')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Bias between FST 12 - 56
    biases_mean = np.array(fst12_means) - np.array(fst56_means)
    bias_mins = np.array(fst12_mins) - np.array(fst56_maxs)
    bias_maxs = np.array(fst12_maxs) - np.array(fst56_mins)
    bias_ranges = bias_maxs - bias_mins

    ax2.errorbar(shots, biases_mean, yerr=bias_ranges if use_error_bars else None, fmt='o-', color='purple',
                 label=f'Bias (FST 1/2 - 5/6)', linewidth=2, capsize=5)

    ax2.set_xlabel('Number of Shots')
    ax2.set_ylabel('Bias')
    ax2.set_title('Bias Analysis')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    fig_name = f"{label}_{score_type}_by_shots_bias.png"
    fig.savefig(f"Multiple_experiments_plots/{fig_name}", dpi=300, bbox_inches='tight')
    plt.close(fig)


# def plot_experiment_results(bulk_metrics, fst12_metrics, fst56_metrics, score_type, shots, use_error_bars=True, label=""):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Use custom colors for the lines
#     line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

#     # Calculate statistics for each experiment
#     bulk_means = [np.mean([d['mean'][i][score_type] for d in bulk_metrics]) for i in range(len(bulk_metrics[0]['mean']))]
#     bulk_stds = [np.std([d['std'][i][score_type] for d in bulk_metrics]) for i in range(len(bulk_metrics[0]['std']))]
#     bulk_mins = [np.min([d['mean'][i][score_type] for d in bulk_metrics]) for i in range(len(bulk_metrics[0]['mean']))]
#     bulk_maxs = [np.max([d['mean'][i][score_type] for d in bulk_metrics])for i in range(len(bulk_metrics[0]['mean']))]
#     bulk_ranges = [bulk_maxs[i] - bulk_mins[i] for i in range(len(bulk_mins))]

#     fst12_means = [np.mean([d['mean'][i][score_type] for d in fst12_metrics]) for i in range(len(fst12_metrics[0]['mean']))]
#     fst12_stds = [np.std([d['std'][i][score_type] for d in fst12_metrics]) for i in range(len(fst12_metrics[0]['std']))]
#     fst12_mins = [np.min([d['mean'][i][score_type] for d in fst12_metrics]) for i in range(len(fst12_metrics[0]['mean']))]
#     fst12_maxs = [np.max([d['mean'][i][score_type] for d in fst12_metrics]) for i in range(len(fst12_metrics[0]['mean']))]
#     fst12_ranges = [fst12_maxs[i] - fst12_mins[i] for i in range(len(fst12_mins))]

#     fst56_means = [np.mean([d['mean'][i][score_type] for d in fst56_metrics]) for i in range(len(fst56_metrics[0]['mean']))]
#     fst56_stds = [np.std([d['std'][i][score_type] for d in fst56_metrics]) for i in range(len(fst56_metrics[0]['std']))]
#     fst56_mins = [np.min([d['mean'][i][score_type] for d in fst56_metrics]) for i in range(len(fst56_metrics[0]['mean']))]
#     fst56_maxs = [np.max([d['mean'][i][score_type] for d in fst56_metrics]) for i in range(len(fst56_metrics[0]['mean']))]
#     fst56_ranges = [fst56_maxs[i] - fst56_mins[i] for i in range(len(fst56_mins))]

#     print(bulk_means, bulk_ranges, fst12_means, fst12_ranges, fst56_means, fst56_ranges)
    
#     # Plot the results
#     ax1.errorbar(shots, bulk_means, yerr=bulk_ranges, fmt='o-', 
#                 color=line_colors[0], label=f'Aggregate {score_type}', 
#                 linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
#     ax1.errorbar(shots, fst12_means, yerr=fst12_ranges, fmt='s-',
#                 color=line_colors[1], label=f'FST 1/2 {score_type}',
#                 linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
#     ax1.errorbar(shots, fst56_means, yerr=fst56_ranges, fmt='^-',
#                 color=line_colors[2], label=f'FST 5/6 {score_type}',
#                 linewidth=2, capsize=5, capthick=1, elinewidth=1)
    
#     ax1.set_xlabel('Number of Shots')
#     ax1.set_ylabel(f'{score_type}')
#     ax1.set_title(f'{score_type} by Number of Shots and Skin Type')
#     ax1.legend()
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot bias with confidence intervals
#     biases_mean = np.array(fst12_means) - np.array(fst56_means)
#     bias_mins = np.array(fst12_mins) - np.array(fst56_maxs)
#     bias_maxs = np.array(fst12_maxs) - np.array(fst56_mins)
#     bias_ranges = bias_maxs - bias_mins

#     print(biases_mean, bias_ranges)

#     ax2.errorbar(shots, biases_mean, yerr=bias_ranges, fmt='o-',
#                 color='purple', label=f'Bias (FST 1/2 - FST 5/6 {score_type})',
#                 linewidth=2, capsize=5, capthick=1, elinewidth=1)

#     ax2.set_xlabel('Number of Shots')
#     ax2.set_ylabel('Bias')
#     ax2.set_title('Bias by Number of Shots')
#     ax2.legend()
#     ax2.grid(True, linestyle='--', alpha=0.7)
    
#     fig_name = f"{label}_{score_type}_by_shots_and_skin_type"
#     fig.savefig("Multiple_experiments_plots/" + fig_name + ".png", dpi=300, bbox_inches='tight')
#     plt.close(fig)

def parse_experiment_name(filename, model_name="gpt", num_shots=50):
    """Parses the experiment name to extract relevant information."""
    normalized_filename = filename.replace("\\", "/")
    pattern = (
        rf"ddi_results/[^/]+/ddi_(?P<fst12_ben>\d+)_(?P<fst12_mal>\d+)_"
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
        if experiment_info is None:
            continue
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
    exps = Path('./ddi_results').rglob('*.pkl')  # find all pickle files in results folder and subdirs
    exps = [str(exp) for exp in exps]
    # Loop through each experiment
    for model in models:
        model_exps = [exp for exp in exps if model in exp]
        sorted_experiments = sort_experiments(model_exps, model_name=model)
        if len(sorted_experiments) == 0:
            print(f"No experiments found for model: {model}")
            continue
    
        for category, (filenames, shots) in sorted_experiments.items():
            shots = list(set(shots))[:-1]
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
                if "90" in filename:
                    continue
                subdir = Path(filename).parent.name
                filenames_by_subdir[subdir].append(filename)
            for subdir, subdir_filenames in filenames_by_subdir.items():
                bulk_metrics, fst12_metrics, fst56_metrics = calculate_bootstrap_metrics(subdir_filenames)
                overall_bulk_metrics.append(bulk_metrics)
                overall_fst12_metrics.append(fst12_metrics)
                overall_fst56_metrics.append(fst56_metrics)
            print(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "accuracy", shots, use_error_bars=True, label=category) # overall_bulk_metrics, fst12_metrics, fst56_metrics, "accuracy", shots, use_error_bars=True, label=category)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "tpr", shots, use_error_bars=True, label=category)
            plot_experiment_results(overall_bulk_metrics, overall_fst12_metrics, overall_fst56_metrics, "fscore", shots, use_error_bars=True, label=category)