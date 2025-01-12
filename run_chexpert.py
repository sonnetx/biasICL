import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI
import pandas as pd


def create_demo(white_ben, white_mal, black_ben, black_mal):
    ###
    ### Load demo example frame
    ### Choose relevant demo examples
    ### Then create demo prompt and list of demo image paths
    ###
    dataset_name = "chexpert_binary_PNA"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/demo.csv", index_col=0)
    total_samples = white_ben + white_mal + black_ben + black_mal
    
    black_frame = demo_frame[demo_frame.binary_race == "Black"]
    black_mal_frame = black_frame[black_frame.Pneumonia == True].sample(black_mal, random_state=141)
    black_ben_frame = black_frame[black_frame.Pneumonia == False].sample(black_ben, random_state=141)
    
    white_frame = demo_frame[demo_frame.binary_race == "White"]
    white_mal_frame = white_frame[white_frame.Pneumonia == True].sample(white_mal, random_state=141)
    white_ben_frame = white_frame[white_frame.Pneumonia == False].sample(white_ben, random_state=141)
    
    final_demo_frame = pd.concat([black_mal_frame,
                                  black_ben_frame,
                                  white_mal_frame,
                                  white_ben_frame]).sample(total_samples, random_state=141) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    white_ben, 
    white_mal, 
    black_ben, 
    black_mal,
    num_qns_per_round,
    detail="auto",
):

    EXP_NAME = f"chexpert_{white_ben}_{white_mal}_{black_ben}_{black_mal}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(white_ben, white_mal, black_ben, black_mal)

    dataset_name = "chexpert_binary_PNA"
    test_df = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/test.csv", index_col=0)

    if model.startswith("gpt"):
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
        ans_choice = "B" if row.Pneumonia == True else "A"
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
        prompt = "Please respond with the following format for each question, in the form of a single capital letter specifying which label best describes the image. Do not deviate from the format, because it will be automatically parsed."
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: Which diagnosis best fits the chest x-ray in the image above?
Choices: A. Pneumonia, B. No Finding
Answer Choice: {demo[1]}
"""
        qns_idx = []
        for idx, row in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(row.Index)
            image_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: Which diagnosis best fits the chest x-ray in the image above?
Choices {qn_idx}: A. Pneumonia, B. No Finding

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
    with open(f"./chexpert_results/{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    results_csv_path = os.path.join(os.getcwd(), f"{dataset_name}_{model}_{num_qns_per_round}_results.csv")
    if not os.path.isfile(results_csv_path):
        # Create a new CSV file with the specified columns
        columns = [
            'num_shots_per_class',  # Number of shots per class
            'black_race_split',  # Proportion of black examples in the malignant class
            'accuracy',  # Accuracy of the model
            'acc_error',  # Error of the accuracy (std dev of the mean)
            'f1',  # F1 score of the model
            'f1_error',  # Error of the F1 score (std dev of the mean)
            'black_accuracy',  # Accuracy of the model on black examples
            'black_acc_error',  # Error of the accuracy on black examples (std dev of the mean)
            'black_f1',  # F1 score of the model on black examples
            'black_f1_error',  # Error of the F1 score on black examples (std dev of the mean)
            'white_accuracy',  # Accuracy of the model on white examples
            'white_acc_error',  # Error of the accuracy on white examples (std dev of the mean)
            'white_f1',  # F1 score of the model on white examples
            'white_f1_error',  # Error of the F1 score on white examples (std dev of the mean)
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(results_csv_path, index=False)

        
if __name__ == "__main__":
    # for num_malignant in [1]:
    #     main("claude", 
    #     num_malignant*3, 
    #     num_malignant,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)

    # for num_malignant in [3, 10]:
    #     main("claude",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     50,)

    # for num_malignant in [1,3,5,7,10]:
    #     main("claude", 
    #     0, 
    #     0,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)
    
    for num_malignant in [0]:
        main("gpt-4o-2024-05-13",
        num_malignant*3, 
        num_malignant, 
        0, 
        0,
        50,)

    # for num_malignant in [1,5,10,20,30]:
    #     main("gpt-4o-2024-05-13",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     50,)

    # for num_malignant in [1,5,10,20,30]:
    #     main("gpt-4o-2024-05-13", 
    #     0, 
    #     0,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)
    
    # for num_malignant in [1,5,10,20,30]:
    #     main("gpt-4o-2024-05-13", 
    #     num_malignant*3, 
    #     num_malignant,
    #     num_malignant*3, 
    #     num_malignant,
    #     50,)

    # for num_malignant in [1,5,10,20,30]:
    #     main("Gemini1.5",
    #     num_malignant*3, 
    #     num_malignant, 
    #     0, 
    #     0,
    #     10,)

    #     main("Gemini1.5",
    #     0,
    #     0,
    #     num_malignant*3, 
    #     num_malignant, 
    #     10,)

    #     main("Gemini1.5",
    #     num_malignant*3, 
    #     num_malignant, 
    #     num_malignant*3, 
    #     num_malignant,
    #     10,)