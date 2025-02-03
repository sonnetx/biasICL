import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI
import pandas as pd

def vector_to_letters(vector):
    index_to_letter = {i: chr(i + ord('A')) for i, val in enumerate(vector) if val == 1}
    letters = [index_to_letter[i] for i, val in enumerate(vector) if val == 1]
    return ', '.join(letters)

def create_demo(male_number_shots, female_number_shots):

    demo_frame = pd.read_csv("/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/demo.csv", index_col=0)
    total_samples = male_number_shots + female_number_shots
    
    male_frame = demo_frame[demo_frame.Sex == 'Male']
    female_frame = demo_frame[demo_frame.Sex == 'Female']
    
    # logic for sampling k positive shots and k negative shots for each label 
    individual_frames = []
    for i in range(5,19):
        # male positive examples
        individual_frames.append(male_frame[male_frame.iloc[:,i] == 1].sample(male_number_shots, random_state=42))
        # male negative examples
        individual_frames.append(male_frame[male_frame.iloc[:,i] != 1].sample(male_number_shots, random_state=42))
        # female positive examples
        individual_frames.append(female_frame[female_frame.iloc[:,i] == 1].sample(female_number_shots, random_state=42))
        # female negative examples
        individual_frames.append(female_frame[female_frame.iloc[:,i] != 1].sample(female_number_shots, random_state=42))
    
    final_demo_frame = pd.concat(individual_frames).sample(total_samples, random_state=42) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    male_number_shots, 
    female_number_shots,
    num_qns_per_round,
    detail="low",
):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.

    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    male_number_shots[int]: number of demonstrating examples (pos and neg) to include from Male patients
    female_number_shots[int]: number of demonstrating examples (pos and neg) to include from Female patients
    num_qns_per_round[int]: number of queries to be batched in one API call
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    """

    base_dir = '/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/'
    EXP_NAME = f"chexpertMultilabel_{male_number_shots}_{female_number_shots}_{model}_{num_qns_per_round}"
    
    demo_frame = create_demo(male_number_shots, female_number_shots)
    test_df = pd.read_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/test.csv', index_col=0)

    if model.startswith("gpt"):
        api = GPT4VAPI(model=model, detail=detail)
    else:
        assert model == "Gemini1.5"
        api = GeminiAPI()
    print(EXP_NAME, f"test size = {len(test_df)}")
    
    # create demo_examples from my demo_frame
    # list of tuples
    # i[0] = (path_to_image, class name)
    demo_paths = []
    demo_labels = []
    for i,row in demo_frame.iterrows():
        ground_truth_vec = (row.iloc[5:19] == 1).values.astype(float)
        demo_paths.append(base_dir+row.Path)
        demo_labels.append(ground_truth_vec)
    demo_examples = list(zip(demo_paths, demo_labels))
    
    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(demo_frame.columns[5:19])])
    test_df = test_df.sample(frac=1, random_state=66)  # Shuffle the test set

    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)

        prompt = ""
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            ground_truth_vec = demo[1]
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: What is in the image above?
Choices: {formatted_columns}
Answer: <ANS> {vector_to_letters(ground_truth_vec)} </ANS>
"""
        qns_idx = []
        for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(i.Index)
            image_paths.append(base_dir+ i.Path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question {qn_idx}: What is in the image above?
Choices {qn_idx}: {formatted_columns}

"""
        for i in range(start_idx, end_idx):
            qn_idx = i - start_idx + 1
            prompt += f"""
Please respond with the following format for each question:
---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
Answer {qn_idx}: <ANS> Your comma-delimited list of capital letters representing radiographic findings here </ANS>
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
    with open(f"{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)


        
if __name__ == "__main__":
    main("gpt-4o-2024-05-13",
    0, 
    0,
    50,)

    main("gpt-4o-2024-05-13",
    2, 
    2,
    50,)