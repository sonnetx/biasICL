import pandas as pd
import re
from pathlib import Path

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
        if experiment_info:
            if experiment_info["fst12_ben"] == experiment_info["fst56_ben"] and experiment_info["fst12_mal"] == experiment_info["fst56_mal"]:
                experiments["matched_fst12_fst56"].append(filename)
            elif experiment_info["fst12_ben"] == experiment_info["fst56_ben"] and experiment_info["fst12_mal"] == experiment_info["fst56_mal"] and experiment_info["fst12_ben"] == experiment_info["fst12_mal"]:
                experiments["matched_balanced_fst12_fst56"].append(filename)
            elif experiment_info["fst56_ben"] == 0 and experiment_info["fst56_mal"] == 0:
                experiments["fst12_only"].append(filename)
            elif experiment_info["fst12_ben"] == 0 and experiment_info["fst12_mal"] == 0:
                experiments["fst56_only"].append(filename)
            elif experiment_info["fst12_ben"] == experiment_info["fst12_mal"]:
                experiments["balanced_fst12_only"].append(filename)
            elif experiment_info["fst56_ben"] == experiment_info["fst56_mal"]:
                experiments["balanced_fst56_only"].append(filename)

    # Sort experiments within each category
    for category, filenames in experiments.items():
        def sort_key(filename):
            experiment_info = parse_experiment_name(filename, model_name=model_name, num_shots=num_shots)
            if experiment_info["fst12_ben"] == 0:
                return experiment_info["fst56_ben"]
            else:
                return experiment_info["fst12_ben"]
        experiments[category] = sorted(filenames, key=sort_key)

    return experiments

def create_demo(fst12_ben, fst12_mal, fst56_ben, fst56_mal):
    ###
    ### Load demo example frame
    ### Choose relevant demo examples
    ### Then create demo prompt and list of demo image paths
    ###
    dataset_name = "DDI"
    demo_frame = pd.read_csv(f"C:/Users/sonne/Documents/Github/biasICL/ddi_demo_metadata.csv", index_col=0)
    total_samples = fst12_ben + fst12_mal + fst56_ben + fst56_mal
    
    fst56_frame = demo_frame[demo_frame.skin_tone == 56]
    print("fst56_frame malignant", len(fst56_frame[fst56_frame.malignant == True]))
    print("fst56_frame benign", len(fst56_frame[fst56_frame.malignant == False]))
    fst56_mal_frame = fst56_frame[fst56_frame.malignant == True].sample(fst56_mal, random_state=42)
    fst56_ben_frame = fst56_frame[fst56_frame.malignant == False].sample(fst56_ben, random_state=42)
    
    fst12_frame = demo_frame[demo_frame.skin_tone == 12]
    print("fst12_frame malignant", len(fst12_frame[fst12_frame.malignant == True]))
    print("fst12_frame benign", len(fst12_frame[fst12_frame.malignant == False]))
    fst12_mal_frame = fst12_frame[fst12_frame.malignant == True].sample(fst12_mal, random_state=42)
    fst12_ben_frame = fst12_frame[fst12_frame.malignant == False].sample(fst12_ben, random_state=42)
    
    final_demo_frame = pd.concat([fst56_mal_frame,
                                  fst56_ben_frame,
                                  fst12_mal_frame,
                                  fst12_ben_frame]).sample(total_samples, random_state=42) # sample full num to shuffle
    return final_demo_frame

models = ["gpt-4o-2024-05-13"]
exps = Path('./ddi_results').glob('*.pkl')  # find all pickle files in results folder
exps = [str(exp) for exp in exps]
print(len(exps), exps)
exps = ['ddi_results/ddi_3_1_3_1_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_15_5_15_5_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_60_20_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_30_10_30_10_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_3_1_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_1_1_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_20_20_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_90_30_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_20_20_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_30_10_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_5_5_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_30_30_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_60_20_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_10_10_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_30_10_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_20_20_20_20_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_10_10_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_60_20_60_20_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_15_5_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_5_5_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_3_1_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_15_5_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_30_30_30_30_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_1_1_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_10_10_10_10_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_5_5_5_5_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_0_0_90_30_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_30_30_0_0_gpt-4o-2024-05-13_50.pkl', 'ddi_results/ddi_1_1_1_1_gpt-4o-2024-05-13_50.pkl']
# Loop through each experiment
for model in models:
    model_exps = [exp for exp in exps if model in exp]
    print(len(model_exps), model_exps)
    sorted_experiments = sort_experiments(model_exps, model_name=model)
    print(len(sorted_experiments), sorted_experiments)
    if len(sorted_experiments) == 0:
        print(f"No experiments found for model: {model}")
        continue

    for category, filenames in sorted_experiments.items():
        print(f"Processing category: {category}")
        print(len(filenames), filenames)