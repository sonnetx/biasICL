import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI
import pandas as pd


def create_demo(female_ben, female_mal, male_ben, male_mal, random_seed=141):
    ###
    ### Load demo example frame
    ### Choose relevant demo examples
    ### Then create demo prompt and list of demo image paths
    ###
    dataset_name = "chexpert_binary_PNA"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/demo.csv", index_col=0)
    total_samples = female_ben + female_mal + male_ben + male_mal
    
    female_frame = demo_frame[demo_frame.Sex == "Female"]
    if len(female_frame[female_frame.Pneumothorax == True]) < female_mal:
        print(f"Warning: not enough female malignant samples, taking the max available {len(female_frame[female_frame.Pneumothorax == True])}")
        female_mal_frame = female_frame[female_frame.Pneumothorax == True].sample(len(female_frame[female_frame.Pneumothorax == True]), random_state=random_seed)
    else:
        female_mal_frame = female_frame[female_frame.Pneumothorax == True].sample(female_mal, random_state=random_seed)
    
    if len(female_frame[female_frame.Pneumothorax == False]) < female_ben:
        print(f"Warning: not enough female benign samples, taking the max available {len(female_frame[female_frame.Pneumothorax == False])}")
        female_ben_frame = female_frame[female_frame.Pneumothorax == False].sample(len(female_frame[female_frame.Pneumothorax == False]), random_state=random_seed)
    else:
        female_ben_frame = female_frame[female_frame.Pneumothorax == False].sample(female_ben, random_state=random_seed)
    
    male_frame = demo_frame[demo_frame.Sex == "Male"]
    if len(male_frame[male_frame.Pneumothorax == True]) < male_mal:
        print(f"Warning: not enough male malignant samples, taking the max available {len(male_frame[male_frame.Pneumothorax == True])}")
        male_mal_frame = male_frame[male_frame.Pneumothorax == True].sample(len(male_frame[male_frame.Pneumothorax == True]), random_state=random_seed)
    else:
        male_mal_frame = male_frame[male_frame.Pneumothorax == True].sample(male_mal, random_state=random_seed)
    
    if len(male_frame[male_frame.Pneumothorax == False]) < male_ben:
        print(f"Warning: not enough male benign samples, taking the max available {len(male_frame[male_frame.Pneumothorax == False])}")
        male_ben_frame = male_frame[male_frame.Pneumothorax == False].sample(len(male_frame[male_frame.Pneumothorax == False]), random_state=random_seed)
    else:
        male_ben_frame = male_frame[male_frame.Pneumothorax == False].sample(male_ben, random_state=random_seed)
    
    total_samples = len(female_mal_frame) + len(female_ben_frame) + len(male_mal_frame) + len(male_ben_frame)
    final_demo_frame = pd.concat([female_mal_frame,
                                  female_ben_frame,
                                  male_mal_frame,
                                  male_ben_frame]).sample(total_samples, random_state=random_seed) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    female_ben, 
    female_mal, 
    male_ben, 
    male_mal,
    num_qns_per_round,
    detail="auto",
    random_seed=141
):

    EXP_NAME = f"chexpert_{female_ben}_{female_mal}_{male_ben}_{male_mal}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(female_ben, female_mal, male_ben, male_mal, random_seed=random_seed)

    dataset_name = "chexpert_binary_PNA"
    test_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/test.csv", index_col=0)

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
        ans_choice = "B" if row.Pneumothorax == True else "A"
        demo_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
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
                Question: Which diagnosis best fits the chest x-ray in the image above?
                Choices: A. Pneumothorax, B. No Finding
                Answer Choice: {demo[1]}
                """
        qns_idx = []
        for idx, row in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(row.Index)
            image_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
                Question {qn_idx}: Which diagnosis best fits the chest x-ray in the image above?
                Choices {qn_idx}: A. Pneumothorax, B. No Finding

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
    with open(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/chexpert_results/base_rate_exps/{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)

        
if __name__ == "__main__":

    # main("claude", 
    #     0, 
    #     0,
    #     0, 
    #     0,
    #     50,)
    
    # for num_malignant in [1,3,5,7,10]:
    #     main("claude", 
    #     num_malignant*3, 
    #     num_malignant,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)

    #     main("claude",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     50,)

    #     main("claude", 
    #     0, 
    #     0,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)

    # test the base rate
    for model in ["Gemini1.5", "gpt-4o-2024-05-13", "claude"]:
        for seed in [10, 100, 141]:
            main(model,
                0, 
                0, 
                0, 
                0,
                50, 
                random_seed=seed)

            # base rate experiments
            total = 300
            for i in range(0, total, 10):
                main(model, i, total - i, total - i, i, 50, random_seed=seed)

            for i in range(0, total, 10):
                main(model, total - i, i, total - i, i, 50, random_seed=seed)
    
    # for num_malignant in [1,5,10,20,30]:
    #     main("gpt-4o-2024-05-13",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     50,)

    #     main("gpt-4o-2024-05-13", 
    #     0, 
    #     0,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)

    #     main("gpt-4o-2024-05-13", 
    #     num_malignant*3, 
    #     num_malignant,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)


    #     main("Gemini1.5",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     50,)

    #     main("Gemini1.5",
    #     0,
    #     0,
    #     num_malignant*3, 
    #     num_malignant, 
    #     50,)

    #     main("Gemini1.5",
    #     num_malignant*3, 
    #     num_malignant, 
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)