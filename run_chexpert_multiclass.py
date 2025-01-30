import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI
import pandas as pd
import re

def parse_answers(text):
    pattern = r"<ANS>\s*([A-N, ]+)\s*</ANS>"
    match = re.search(pattern, text)
    letter_to_index = {chr(i): i - ord('A') for i in range(ord('A'), ord('N') + 1)}
    vector = [0] * 14
    
    if match:
        letters = match.group(1).split(',')
        for letter in letters:
            letter = letter.strip()
            if letter in letter_to_index:
                vector[letter_to_index[letter]] = 1
    
    return vector

def create_demo(female_count, male_count):
    dataset_name = "chexpert_binary_PNA"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/demo.csv", index_col=0)
    total_samples = female_count + male_count
    
    female_frame = demo_frame[demo_frame.Sex == "Female"]
    
    male_frame = demo_frame[demo_frame.Sex == "Male"]

    final_demo_frame = pd.concat([female_frame.sample(female_count, random_state=141),
                                  male_frame.sample(male_count, random_state=141)]).sample(total_samples, random_state=141) # sample full num to shuffle
    return final_demo_frame

def main(
    model,
    female_count,
    male_count,
    num_qns_per_round,
    detail="auto",
):

    EXP_NAME = f"chexpert_{female_count}_{male_count}_{model}_{num_qns_per_round}"
    
    # demo_frame = create_demo(female_count, male_count)

    dataset_name = "chexpert_binary_PNA"
    demo_frame = pd.read_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/{dataset_name}/demo.csv", index_col=0)
    
    labels = demo_frame.columns[5:19]
    total_samples = female_count + male_count
    demo_frames = []
    num_shots = total_samples // len(labels)
    
    # Keep track of sampled indices
    sampled_indices = set()

    for k in range(num_shots):
        for L in labels:
            # Count how many samples we already have for this label
            existing_samples = sum(1 for idx in sampled_indices 
                                if demo_frame.loc[idx, L] == 1)
            
            # Calculate how many more samples we need
            samples_needed = k - existing_samples
            
            if samples_needed > 0:
                # Get eligible indices (excluding already sampled ones)
                eligible = demo_frame[
                    (demo_frame[L] == 1) & 
                    (~demo_frame.index.isin(sampled_indices))
                ]
                
                # Sample the remaining needed samples
                new_samples = eligible.sample(min(samples_needed, len(eligible)))
                sampled_indices.update(new_samples.index)
                demo_frames.append(new_samples)
            
    demo_frame = pd.concat(demo_frames).sample(total_samples, random_state=141) # sample full num to shuffle
    
    print("Composition of demo frame:")
    for L in labels:
        print(f"{L}: {sum(demo_frame[L] == 1)}")

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
        demo_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
        formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(row.index[5:19])])
        demo_labels.append(formatted_columns)
    demo_examples = list(zip(demo_paths, demo_labels))
    
    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}
    new_results = []

    test_df = test_df.sample(frac=1, random_state=141)  # Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)
        row = test_df.iloc[start_idx]

        # race = 'White' if row.race in ['White', 'White, non-Hispanic', 'White or Caucasian'] else 'Not White'
        age = row.Age
        sex = row.Sex
        ground_truth_vec = (row.iloc[5:19] == 1).values.astype(float)

        formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(row.index[5:19])])
        prompt = "Please respond with the following format for each question. Do not deviate from the format, because it will be automatically parsed."
        image_paths = [
            i[0] for i in demo_examples
        ]
        for demo in demo_examples:
            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
            Question: Which of the following radiographic findings are present in the image above? More than one finding may be present per image.
            Choices: {formatted_columns}
            Answer Choice: {demo[1]}
            """
        qns_idx = []
        for idx, row in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
            qns_idx.append(row.Index)
            image_paths.append('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/' + row.Path)
            qn_idx = idx + 1

            prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
            Question: Which of the following radiographic findings are present in the image above? More than one finding may be present per image.
            Choices: {formatted_columns}

            """
        for i in range(start_idx, end_idx):
            qn_idx = i - start_idx + 1
            prompt += f"""
                    Please respond with the following format for each question, in the form of a comma delimited list of capital letters specifying which radiographic findings are present in the image surrounded by beginning <ANS> and end </ANS> brackets:
                        ---BEGIN FORMAT TEMPLATE FOR QUESTION---
                        <ANS> Your comma-delimited list of capital letters representing radiographic findings here </ANS>
                        ---END FORMAT TEMPLATE FOR QUESTION---

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
            ans = parse_answers(res)
            new_results.append({
                "response": res,
                "parsed_answer": ans,
                "age": age,
                "sex": sex,
                "ground_truth": ground_truth_vec,
            })
            results[qns_id] = (res,prompt,image_paths)

    # Update token usage and save the results
    previous_usage = results.get("token_usage", (0, 0, 0))
    total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    results["token_usage"] = total_usage
    with open(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/chexpert_results/{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)
    result_df = pd.DataFrame(new_results)
    result_df.to_csv(f"/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/chexpert_results/{EXP_NAME}.csv")

        
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
    
    total = 100

    for count in range(0, total, 10):
        # only men
        main("gpt-4o-2024-05-13",
            0,
            count, 
            50,)
        
        # only women
        main("gpt-4o-2024-05-13",
            count, 
            0, 
            50,)
        
        # both
        main("gpt-4o-2024-05-13",
            count, 
            count, 
            50,)

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

    # main("Gemini1.5",
    #     0, 
    #     0, 
    #     0, 
    #     0,
    #     50,)

    # for num_malignant in [1,5,10,20,30]:
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