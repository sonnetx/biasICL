import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI
import pandas as pd

rare_diseases = {
        'subcutaneous-t-cell-lymphoma', 'focal-acral-hyperkeratosis', 
        'eccrine-poroma', 'inverted-follicular-keratosis', 'kaposi-sarcoma',
        'metastatic-carcinoma', 'mycosis-fungoides', 
        'acquired-digital-fibrokeratoma', 'atypical-spindle-cell-nevus-of-reed',
        'verruciform-xanthoma', 'morphea', 'nevus-lipomatosus-superficialis',
        'pigmented-spindle-cell-nevus-of-reed', 'arteriovenous-hemangioma',
        'syringocystadenoma-papilliferum', 'trichofolliculoma',
        'coccidioidomycosis', 'leukemia-cutis', 'sebaceous-carcinoma',
        'blastic-plasmacytoid-dendritic-cell-neoplasm', 'glomangioma',
        'dermatomyositis', 'cellular-neurothekeoma', 'graft-vs-host-disease',
        'xanthograngioma', 'chondroid-syringoma', 'angioleiomyoma'
    }

def create_demo(fst12_ben, fst12_mal, fst56_ben, fst56_mal, filter_rare = False, random_seed=141):
    ###
    ### Load demo example frame
    ### Choose relevant demo examples
    ### Then create demo prompt and list of demo image paths
    ###
    dataset_name = "DDI"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/ddi_demo_metadata.csv", index_col=0)
    if filter_rare:
        demo_frame = demo_frame[~demo_frame.disease.isin(rare_diseases)]
    
    total_samples = fst12_ben + fst12_mal + fst56_ben + fst56_mal
    
    fst56_frame = demo_frame[demo_frame.skin_tone == 56]
    fst12_frame = demo_frame[demo_frame.skin_tone == 12]

    print("fst56_frame malignant", len(fst56_frame[fst56_frame.malignant == True]))
    print("fst56_frame benign", len(fst56_frame[fst56_frame.malignant == False]))
    print("fst12_frame malignant", len(fst12_frame[fst12_frame.malignant == True]))
    print("fst12_frame benign", len(fst12_frame[fst12_frame.malignant == False]))

    fst56_mal_frame = fst56_frame[fst56_frame.malignant == True]
    fst56_ben_frame = fst56_frame[fst56_frame.malignant == False]
    
    fst12_mal_frame = fst12_frame[fst12_frame.malignant == True]
    fst12_ben_frame = fst12_frame[fst12_frame.malignant == False]
    
    if len(fst56_mal_frame) < fst56_mal:
        print(f"Warning: not enough malignant samples for skin tone 56, taking the max available {len(fst56_mal_frame)}")
        fst56_mal_frame = fst56_mal_frame.sample(len(fst56_mal_frame), random_state=random_seed)
    else:
        fst56_mal_frame = fst56_mal_frame.sample(fst56_mal, random_state=random_seed)
    
    if len(fst56_ben_frame) < fst56_ben:
        print(f"Warning: not enough benign samples for skin tone 56, taking the max available {len(fst56_ben_frame)}")
        fst56_ben_frame = fst56_ben_frame.sample(len(fst56_ben_frame), random_state=random_seed)
    else:
        fst56_ben_frame = fst56_ben_frame.sample(fst56_ben, random_state=random_seed)
    
    if len(fst12_mal_frame) < fst12_mal:
        print(f"Warning: not enough malignant samples for skin tone 12, taking the max available {len(fst12_mal_frame)}")
        fst12_mal_frame = fst12_mal_frame.sample(len(fst12_mal_frame), random_state=random_seed)
    else:
        fst12_mal_frame = fst12_mal_frame.sample(fst12_mal, random_state=random_seed)
    
    if len(fst12_ben_frame) < fst12_ben:
        print(f"Warning: not enough benign samples for skin tone 12, taking the max available {len(fst12_ben_frame)}")
        fst12_ben_frame = fst12_ben_frame.sample(len(fst12_ben_frame), random_state=random_seed)
    else:
        fst12_ben_frame = fst12_ben_frame.sample(fst12_ben, random_state=random_seed)
    
    final_demo_frame = pd.concat([fst56_mal_frame,
                                      fst56_ben_frame,
                                      fst12_mal_frame,
                                      fst12_ben_frame])
    
    if len(final_demo_frame) < total_samples:
        print(f"Warning: not enough total samples, taking the max available {len(final_demo_frame)}")
        final_demo_frame = final_demo_frame.sample(len(final_demo_frame), random_state=random_seed)
    else:
        final_demo_frame = final_demo_frame.sample(total_samples, random_state=random_seed) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    fst12_ben, 
    fst12_mal, 
    fst56_ben, 
    fst56_mal,
    num_qns_per_round,
    filter_rare = False,
    detail="auto",
    random_seed=42
):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.

    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    fst12_ben[int]: number of demonstrating examples to include from Fitzpatrick Skin Types I/II from class benign
    fst12_mal[int]: number of demonstrating examples to include from Fitzpatrick Skin Types I/II from class malignant
    fst56_ben[int]: number of demonstrating examples to include from Fitzpatrick Skin Types V/VI from class benign
    fst56_mal[int]: number of demonstrating examples to include from Fitzpatrick Skin Types V/VI from class malignant
    location[str]: Vertex AI location e.g. "us-central1","us-west1", not used for GPT-series models
    num_qns_per_round[int]: number of queries to be batched in one API call
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    """

#     class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    EXP_NAME = f"ddi_{random_seed}_{fst12_ben}_{fst12_mal}_{fst56_ben}_{fst56_mal}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(fst12_ben, fst12_mal, fst56_ben, fst56_mal, random_seed=random_seed)

    dataset_name = "DDI"
    test_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/ddi_test_metadata.csv", index_col=0)
    if filter_rare: 
        test_df = test_df[~test_df.disease.isin(rare_diseases)]

    if model.startswith("gpt") or model.startswith("o1"):
        api = GPT4VAPI(model=model, detail=detail)
    elif model.startswith("Gemini"):
        assert model == "Gemini1.5"
        api = GeminiAPI()
    elif model.startswith("claude"):
        api = ClaudeAPI()
    print(EXP_NAME, f"test size = {len(test_df)}")

    # create demo_examples from my demo_frame
    # list of tuples
    # i[0] = (path_to_image, class name)
    demo_paths = []
    demo_labels = []
    for i,row in demo_frame.iterrows():
        ans_choice = "B" if row.malignant == True else "A"
        demo_paths.append('/home/groups/roxanad/ddi/' + row.DDI_file)
        demo_labels.append(ans_choice)
    demo_examples = list(zip(demo_paths, demo_labels))
    
    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=random_seed)  # Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)

#         random.shuffle(demo_examples)  already shuffled so don't need to do this 
        prompt = "Please respond with the following format for each question, in the form of a single capital letter specifying which label best describes the image. Do not deviate from the format, because it will be automatically parsed."
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: Which description best fits the skin lesion in the image above?
Choices: A. Benign, B. Malignant
Answer Choice: {demo[1]}
"""
        qns_idx = []
        for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(i.Index)
            image_paths.append('/home/groups/roxanad/ddi/' + i.DDI_file)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: Which description best fits the skin lesion in the image above?
Choices {qn_idx}: A. Benign, B. Malignant

"""
        for i in range(start_idx, end_idx):
            qn_idx = i - start_idx + 1
            prompt += f"""
Please respond with the following format for each question:
---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

Do not deviate from the above format. Repeat the format template for the answer."""
        qns_id = str(qns_idx)
        for retry in range(3):
            if (
                (qns_id in results)
                and (not results[qns_id][0].startswith("ERROR"))
                and (
                    f"END FORMAT TEMPLATE FOR QUESTION {end_idx-start_idx}"
                    in results[qns_id][0]
                )
            ):  # Skip if results exist and successful
                continue

            try:
                res = api(
                    prompt,
                    image_paths=image_paths,
                    real_call=True,
                    max_tokens=60 * num_qns_per_round,
                )
            except Exception as e:
                res = f"ERROR!!!! {traceback.format_exc()}"
            except KeyboardInterrupt:
                previous_usage = results.get("token_usage", (0, 0, 0))
                total_usage = tuple(
                    a + b for a, b in zip(previous_usage, api.token_usage)
                )
                results["token_usage"] = total_usage
                with open(f"{EXP_NAME}.pkl", "wb") as f:
                    pickle.dump(results, f)
                exit()

            print(res)
            results[qns_id] = (res,prompt,image_paths)

    # Update token usage and save the results
    previous_usage = results.get("token_usage", (0, 0, 0))
    total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    results["token_usage"] = total_usage
    with open(f"./ddi_results/{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)

        
if __name__ == "__main__":

    # test the base rate
    for model in ["gpt-4o-2024-05-13", "claude"]:
        for seed in [10, 100, 141]:
            main(model,
                40, 0, 40, 0, 50, random_seed=seed)
            
            main(model,
                30, 10, 30, 10, 50, random_seed=seed)
            
            main(model,
                20, 20, 20, 20, 50, random_seed=seed)
            
            main(model,
                10, 30, 10, 30, 50, random_seed=seed)
            
            main(model,
                0, 40, 0, 40, 50, random_seed=seed)
            
            # inverted base rate
            
            main(model,
                0, 40, 40, 0, 50, random_seed=seed)
            
            main(model,
                10, 30, 30, 10, 50, random_seed=seed)
            
            main(model,
                30, 10, 10, 30, 50, random_seed=seed)
            
            main(model,
                40, 0, 0, 40, 50, random_seed=seed)

    # normal ICL experiments
    for model in ["Gemini1.5", "gpt-4o-2024-05-13", "claude"]:
        for seed in [10, 100, 141]:     
            main(model, 
                0,  
                0, 
                0, 
                0,
                50, random_seed=seed)

            for num_malignant in [1,3,5,6]:
                main(model, 
                num_malignant*3, 
                num_malignant,
                num_malignant*3, 
                num_malignant,
                50, random_seed=seed)

                main(model,
                num_malignant*3, 
                num_malignant, 
                0, 
                0,
                50, random_seed=seed)

                main(model, 
                0, 
                0,
                num_malignant*3, 
                num_malignant,
                50, random_seed=seed)