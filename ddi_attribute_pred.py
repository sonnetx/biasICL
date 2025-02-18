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

def create_demo(fst12, fst56, filter_rare = False, random_seed=141):
    dataset_name = "DDI"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/ddi_demo_metadata.csv", index_col=0)
    if filter_rare:
        demo_frame = demo_frame[~demo_frame.disease.isin(rare_diseases)]
    
    total_samples = fst12 + fst56
    
    fst56_frame = demo_frame[demo_frame.skin_tone == 56]
    if len(fst56_frame) < fst56:
        print(f"Warning: not enough samples for skin tone 56, taking the max available {len(fst56_frame)}")
        fst56_frame = fst56_frame.sample(len(fst56_frame), random_state=random_seed)
    else:
        fst56_frame = fst56_frame.sample(fst56, random_state=random_seed)
    
    fst12_frame = demo_frame[demo_frame.skin_tone == 12]
    if len(fst12_frame) < fst12:
        print(f"Warning: not enough samples for skin tone 12, taking the max available {len(fst12_frame)}")
        fst12_frame = fst12_frame.sample(len(fst12_frame), random_state=random_seed)
    else:
        fst12_frame = fst12_frame.sample(fst12, random_state=random_seed)
    
    
    final_demo_frame = pd.concat([fst56_frame, fst12_frame])
    
    if len(final_demo_frame) < total_samples:
        print(f"Warning: not enough total samples, taking the max available {len(final_demo_frame)}")
        final_demo_frame = final_demo_frame.sample(len(final_demo_frame), random_state=random_seed)
    else:
        final_demo_frame = final_demo_frame.sample(total_samples, random_state=random_seed) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    fst12,
    fst56,
    num_qns_per_round,
    filter_rare = False,
    detail="auto",
    random_seed=141
):

#     class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    EXP_NAME = f"ddi_fst_{fst12}_{fst56}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(fst12, fst56, random_seed=random_seed)

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
        ans_choice = "B" if row.skin_tone == 56 else "A"
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
Question: What skin tone does the person in the image have?
Choices: A. White, B. Black or Brown
Answer Choice: {demo[1]}
"""
        qns_idx = []
        for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(i.Index)
            image_paths.append('/home/groups/roxanad/ddi/' + i.DDI_file)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: What skin tone does the person in the image have?
Choices {qn_idx}: A. White, B. Black or Brown

"""
        for i in range(start_idx, end_idx):
            qn_idx = i - start_idx + 1
            prompt += f"""Please respond with the following format for each question:
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

    # for i in range(0, 40, 8):
    #     # main("gpt-4o-2024-05-13",
    #     #     i, 
    #     #     i, 
    #     #     50,)
        
    #     # main("Gemini1.5",
    #     #     i, i,
    #     #     50,)

    #     main("claude",
    #         i, i,
    #         50,)

    # main("gpt-4o-2024-05-13",
    #     0, 
    #     0, 
    #     50,)
        
    # main("Gemini1.5",
    #     0, 0,
    #     50,)

    # main("claude",
    #     0, 0,
    #     50,)

    for model in ["Gemini1.5", "gpt-4o-2024-05-13", "claude"]:
        for seed in [10, 100, 141]:   
            for num_malignant in [1, 5, 10, 15, 20, 30,]:
                main(model,
                num_malignant, 
                num_malignant, 
                50,
                random_seed=seed)