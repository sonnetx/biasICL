import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI
import pandas as pd


def create_demo(female, male):
    dataset_name = "chexpert_binary_PNA"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/demo.csv", index_col=0)
    total_samples = female + male
    
    female_frame = demo_frame[demo_frame.Sex == "Female"]
    male_frame = demo_frame[demo_frame.Sex == "Male"]
    print("females in demo", len(female_frame), "males in demo", len(male_frame))
    
    if len(female_frame) < female:
        print(f"Warning: not enough female samples for attribute, taking the max available {len(female_frame)}")
        female_frame = female_frame.sample(len(female_frame), random_state=141)
    else:
        female_frame = female_frame.sample(female, random_state=141)
        
    if len(male_frame) < male:
        print(f"Warning: not enough male samples for attribute, taking the max available {len(male_frame)}")
        male_frame = male_frame.sample(len(male_frame), random_state=141)
    else:
        male_frame = male_frame.sample(male, random_state=141)
    
    try:
        final_demo_frame = pd.concat([female_frame,
                                      male_frame]).sample(total_samples, random_state=141) # sample full num to shuffle
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        final_demo_frame = pd.concat([female_frame, male_frame])

    return final_demo_frame

def main(
    model,
    female, 
    male,
    num_qns_per_round,
    detail="auto",
):

    EXP_NAME = f"chexpert_attribute_{female}_{male}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(female, male)

    dataset_name = "chexpert_binary_PNA"
    test_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/test.csv", index_col=0)

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
    demo_paths = []
    demo_labels = []
    for i,row in demo_frame.iterrows():
        ans_choice = "B" if row.Sex == "Male" else "A"
        demo_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
        demo_labels.append(ans_choice)
    demo_examples = list(zip(demo_paths, demo_labels))
    
    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=141)  # Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)

#         random.shuffle(demo_examples)  already shuffled so don't need to do this 
        prompt = "You are an expert radiologist working on a research project helping to \
            determine how demographic characteristics like patient-reported race impact the \
                fairness and safety of computer vision classifiers. Please respond with the \
                following format for each question, in the form of a single capital letter \
                specifying which label best describes the image. Do not deviate from the format, \
                    because it will be automatically parsed."
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            prompt += f"""Given the image above, answer the following question using the specified format. 
Question: What sex is the patient in the radiograph above?
Choices: A. Female B. Male
Answer Choice: {demo[1]}
                """
        qns_idx = []
        for idx, row in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(row.Index)
            image_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: What sex is the patient in the radiograph above?
Choices {qn_idx}: A. Female B. Male
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
    with open(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/chexpert_results/{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)

        
if __name__ == "__main__":    
    main("gpt-4o-2024-05-13",
        0, 0,
        50,)
    
    main("Gemini1.5",
        0, 0,
        50,)

    main("claude",
        0, 0,
        50,)