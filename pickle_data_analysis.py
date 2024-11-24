import pandas as pd, numpy as np
import argparse
import pickle
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.utils import resample

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


def res_to_vec(results):
    test_df = pd.read_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/DDI/ddi_test_metadata.csv', index_col=0)
    num_errors = 0
    labels, preds, race = [], [], []
    print(results)
    for i in test_df.itertuples():
        fst = '12' if i.skin_tone == 12 else '56'
        if (i.Index not in results) or (results[i.Index].startswith("ERROR")):
            num_errors += 1

            print(i.Index, f"answer not found")
            continue

        pred_text = results[i.Index]
        ground_truth = "B" if i.malignant == True else "A"
        labels.append(ground_truth)
        preds.append(pred_text)
        race.append(fst)
        
    # if num_errors != 0:
    #     print(results)
    print(f"In total {num_errors} errors; len = {len(labels)}")
    return labels,preds,race

def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics from the actual and predicted labels.

    Args:
        actual: The actual labels.
        predicted: The predicted labels.

    Returns:
        accuracy: The accuracy of the predictions.
        tpr: The true positive rate (sensitivity).
        tnr: The true negative rate (specificity).
        f1_score: The F1 score of the predictions.
    """
    predicted_choice = [p[0] for p in predicted]

    TP = sum((a == 'B' and p == 'B') for a, p in zip(actual, predicted_choice))
    TN = sum((a == 'A' and p == 'A') for a, p in zip(actual, predicted_choice))
    FP = sum((a == 'A' and p == 'B') for a, p in zip(actual, predicted_choice))
    FN = sum((a == 'B' and p == 'A') for a, p in zip(actual, predicted_choice))
    
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

def filter_by_race(actual, predicted, race, race_value):
    filtered_actual = [a for a, r in zip(actual, race) if r == race_value]
    filtered_predicted = [p for p, r in zip(predicted, race) if r == race_value]
    return filtered_actual, filtered_predicted

def plot_experiment_lines(bulk,fst12,fst56,score_type, labels):
    """
    Plot grouped line plot for bulk accuracy, FST 1/2 accuracy, and FST 5/6 accuracy.
    Also plot bias between FST 1/2 and FST 5/6 accuracy.

    Parameters
    ----------
    bulk : List
        List of bulk accuracy scores.
    fst12 : List
        List of FST 1/2 accuracy scores.
    fst56 : List
        List of FST 5/6 accuracy scores.
    score_type : str
        Type of score to plot (e.g. "Accuracy", "F1 score", etc.).
    labels : List
        List of labels for the x-axis.

    Returns
    -------
    None
    """
    biases = np.array(fst12) - np.array(fst56)
    
    # Plot 1: Grouped line plot for bulk accuracy, FST 1/2 accuracy, and FST 5/6 accuracy
    x = np.arange(len(labels))  # the label locations
    
    # Use custom colors for the lines
    line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(x, bulk, label=f'Aggregate {score_type}', color=line_colors[0])
    ax1.plot(x, fst12, label=f'FST 1/2 {score_type}', color=line_colors[1])
    ax1.plot(x, fst56, label=f'FST 5/6 {score_type}', color=line_colors[2])

    # Add labels, title, and custom ticks
    ax1.set_xlabel('Total number of samples used')
    ax1.set_ylabel(f'{score_type}')
    ax1.set_title(f'{score_type} by Experiment')

    # Set wrapped labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)  # Apply wrapped labels

    ax1.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    plt.savefig(score_type + " lines.png")
    
    # Plot 2: Bias plot
    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x, biases, color='purple', label=f'Bias (FST 1/2 - FST 5/6 {score_type})')
    
    # Add labels, title, and custom ticks
    ax2.set_xlabel('Experiments')
    ax2.set_ylabel('Bias')
    ax2.set_title('Bias by Experiment')

    # Set wrapped labels for second plot
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)  # Apply wrapped labels
    
    ax2.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    plt.savefig('plots/' + score_type + " bias lines.png")

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

def plot_experiment_metrics(bulk,fst12,fst56,score_type, labels):
    """
    Plot grouped bar plot for bulk accuracy, FST 1/2 accuracy, and FST 5/6 accuracy.
    Also plot bias between FST 1/2 and FST 5/6 accuracy.

    Parameters
    ----------
    bulk : List
        List of bulk accuracy scores.
    fst12 : List
        List of FST 1/2 accuracy scores.
    fst56 : List
        List of FST 5/6 accuracy scores.
    score_type : str
        Type of score to plot (e.g. "Accuracy", "F1 score", etc.).
    labels : List
        List of labels for the x-axis.

    Returns
    -------
    None
    """
    biases = np.array(fst12) - np.array(fst56)
    
    # Plot 1: Grouped bar plot for bulk accuracy, FST 1/2 accuracy, and FST 5/6 accuracy
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    # Use custom colors for the bars
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot the zero-shot experiment as a horizontal line for each group
    ax1.axhline(y=bulk[0], color=bar_colors[0], linestyle='--', label='Zero-shot')
    ax1.axhline(y=fst12[0], color=bar_colors[1], linestyle='--', label='Zero-shot FST 1/2')
    ax1.axhline(y=fst56[0], color=bar_colors[2], linestyle='--', label='Zero-shot FST 5/6')
    
    # Plot the other experiments as bars
    rects1 = ax1.bar(x[1:] - width, bulk[1:], width, label=f'Aggregate {score_type}', color=bar_colors[0])
    rects2 = ax1.bar(x[1:], fst12[1:], width, label=f'FST 1/2 {score_type}', color=bar_colors[1])
    rects3 = ax1.bar(x[1:] + width, fst56[1:], width, label=f'FST 5/6 {score_type}', color=bar_colors[2])

    # Add labels, title, and custom ticks
#     ax1.set_ylim([0.30,0.9])
    ax1.set_xlabel('Total number of samples used')
    ax1.set_ylabel(f'{score_type}')
    ax1.set_title(f'{score_type} by Experiment')

    # Set wrapped labels
    ax1.set_xticks(x[1:])
    ax1.set_xticklabels(labels[1:])  # Apply wrapped labels

    ax1.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    plt.savefig(score_type + " bars.png")
    
    # Plot 2: Bias plot
    fig, ax2 = plt.subplots(figsize=(10, 6))
    # Plot the zero-shot experiment as a horizontal line
    ax2.axhline(y=biases[0], color='black', linestyle='--', label='Zero-shot')
    
    # Plot the other experiments as bars
    rects4 = ax2.bar(x[1:], biases[1:], width, color='purple', label=f'Bias (FST 1/2 - FST 5/6 {score_type})')
    
    # Add labels, title, and custom ticks
    ax2.set_xlabel('Experiments')
    ax2.set_ylabel('Bias')
    ax2.set_title('Bias by Experiment')

    # Set wrapped labels for second plot
    ax2.set_xticks(x[1:])
    ax2.set_xticklabels(labels[1:])  # Apply wrapped labels
    
    ax2.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    plt.savefig('plots/' + score_type + " bias bars.png")

def plot_experiment_metrics_with_ci(bulk_metrics, fst12_metrics, fst56_metrics, score_type, shots, use_error_bars=False):
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
    ax1.set_title(f'{score_type} by Number of Shots and Skin Type')
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
    ax2.set_title('Bias by Number of Shots')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig.savefig(f'plots/{score_type}_with_ci.png', dpi=300, bbox_inches='tight')

# Main script
if __name__ == "__main__":
    # models = ["gpt", "Gemini", "claude"]
    models = ["claude"]
    exps = Path('./ddi_results').glob('*.pkl')  # find all pickle files in results folder
    exps = list(exps)
    print(len(exps), exps)
    # Loop through each experiment
    for model in models:
        model_exps = [exp for exp in exps if model in str(exp)]
        print(len(model_exps), model_exps)
        if len(model_exps) == 0:
            print(f"No experiments found for model: {model}")
            continue
        # Initialize metric lists for each group
        all_metrics = {'acc': [], 'tpr': [], 'fscore': [], 'labels': []}

        bulk_accs = []
        bulk_tprs = []
        bulk_fscores = []

        fst12_accs = []
        fst12_tprs = []
        fst12_fscores = []

        fst56_accs = []
        fst56_tprs = []
        fst56_fscores = []

        for i, exp in enumerate(model_exps):
            exp = str(exp)[12:]
            print(exp)
            label = exp.split("_")
            fst12_ben, fst12_mal, fst56_ben, fst56_mal = [int(label[i]) for i in range(1, 5)]
            
            # Create the label
            final_label = ",".join(label[1:5])
            
            # Process results
            results = pickle_to_res(exps[i])
            labels, preds, race = res_to_vec(results)
            
            # Calculate metrics for all data
            if len(labels) == 0 or len(preds) == 0:
                print(f"Skipping experiment {exp} due to zero length labels or preds")
                continue
            accuracy, tpr, tnr, fscore = calculate_metrics(labels, preds)
            
            # Calculate metrics for race '12' and '56'
            actual_12, predicted_12 = filter_by_race(labels, preds, race, '12')
            accuracy_12, tpr_12, tnr_12, fscore_12 = calculate_metrics(actual_12, predicted_12)
            
            actual_56, predicted_56 = filter_by_race(labels, preds, race, '56')
            accuracy_56, tpr_56, tnr_56, fscore_56 = calculate_metrics(actual_56, predicted_56)

            bulk_accs.append(accuracy)
            bulk_tprs.append(tpr)
            bulk_fscores.append(fscore)
            
            fst12_accs.append(accuracy_12)
            fst12_tprs.append(tpr_12)
            fst12_fscores.append(fscore_12)
            
            fst56_accs.append(accuracy_56)
            fst56_tprs.append(tpr_56)
            fst56_fscores.append(fscore_56)

            all_metrics['labels'].append(final_label)
            sorted_labels = sorted(all_metrics['labels'], key=lambda x: (int(x.split(',')[0]) == 0 and int(x.split(',')[1]) == 0 and int(x.split(',')[2]) == 0 and int(x.split(',')[3]) == 0, # 0,0,0,0 first
                                                                            int(x.split(',')[0]) == 0 and int(x.split(',')[1]) == 0, # 0,0,_,_ second
                                                                            int(x.split(',')[2]) == 0 and int(x.split(',')[3]) == 0, # _,_,0,0 third
                                                                            all(int(y) > 0 for y in x.split(',')), # all positive fourth
                                                                            tuple(int(y) for y in x.split(',')))) # then sort by the tuple
            print(sorted_labels)
            
        bulk_metrics, fst12_metrics, fst56_metrics = calculate_bootstrap_metrics(exps)
        shots = [0, 5, 30, 60]
        plot_experiment_metrics_with_ci(bulk_metrics, fst12_metrics, fst56_metrics, "accuracy", shots, use_error_bars=True)
        plot_experiment_metrics_with_ci(bulk_metrics, fst12_metrics, fst56_metrics, "tpr", shots, use_error_bars=True)
        plot_experiment_metrics_with_ci(bulk_metrics, fst12_metrics, fst56_metrics, "fscore", shots, use_error_bars=True)
        
        sorted_bulk_accs = [bulk_accs[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst12_accs = [fst12_accs[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst56_accs = [fst56_accs[all_metrics['labels'].index(x)] for x in sorted_labels]
        plot_experiment_metrics(sorted_bulk_accs, sorted_fst12_accs, sorted_fst56_accs, model + ' Accuracy', sorted_labels)
        
        sorted_bulk_tprs = [bulk_tprs[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst12_tprs = [fst12_tprs[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst56_tprs = [fst56_tprs[all_metrics['labels'].index(x)] for x in sorted_labels]
        plot_experiment_metrics(sorted_bulk_tprs, sorted_fst12_tprs, sorted_fst56_tprs, model + ' TPR', sorted_labels)
        
        sorted_bulk_fscores = [bulk_fscores[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst12_fscores = [fst12_fscores[all_metrics['labels'].index(x)] for x in sorted_labels]
        sorted_fst56_fscores = [fst56_fscores[all_metrics['labels'].index(x)] for x in sorted_labels]
        plot_experiment_metrics(sorted_bulk_fscores, sorted_fst12_fscores, sorted_fst56_fscores, model + ' F1 Score', sorted_labels)


        # # Plot lines
        # plot_experiment_lines(sorted_bulk_accs,sorted_fst12_accs,sorted_fst56_accs, model + ' Accuracy', sorted_labels)
        # plot_experiment_lines(sorted_bulk_tprs,sorted_fst12_tprs,sorted_fst56_tprs, model + ' TPR', sorted_labels)
        # plot_experiment_lines(sorted_bulk_fscores,sorted_fst12_fscores,sorted_fst56_fscores, model + ' F1 Score', sorted_labels)
