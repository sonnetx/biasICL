import time
import pandas as pd
import re
import base64
from LMM import GPT4VAPI, GeminiAPI, ClaudeAPI, OpenAIModel

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

def process_dataframe_iterative(model, test_frame):
    results = []
    for _, row in test_frame.iterrows():
        ground_truth_vec = (row.iloc[5:19] == 1).values.astype(float)
        race = 'White' if row.race in ['White', 'White, non-Hispanic', 'White or Caucasian'] else 'Black'
        age = row.Age
        sex = row.Sex
        path = row.updated_path

        formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(row.index[5:19])])
        prompt = f"""<<IMG>>Given the image above, answer the following question using the specified format. 
        Question: Which of the following radiographic findings are present in the image above? More than one finding may be present per image.
        Choices: {formatted_columns}

        Please respond with the following format for each question, in the form of a comma delimited list of capital letters specifying which radiographic findings are present in the image surrounded by beginning <ANS> and end </ANS> brackets:
        ---BEGIN FORMAT TEMPLATE FOR QUESTION---
        <ANS> Your comma-delimited list of capital letters representing radiographic findings here </ANS>
        ---END FORMAT TEMPLATE FOR QUESTION---

        Do not deviate from the above format. Repeat the format template for the answer."""

        response = model.get_completion(prompt, path, None)
        ans = parse_answers(response)

        results.append({
            "response": response,
            "parsed_answer": ans,
            "age": age,
            "sex": sex,
            "path": path,
            "ground_truth": ground_truth_vec,
            "race": race
        })

    result_df = pd.DataFrame(results)
    return result_df

def main():
    test_frame = pd.read_csv('/home/groups/roxanad/sonnet/icl/ManyICL/ManyICL/dataset/chexpert_binary_PNA/test.csv', index_col=0)
    test_frame_first_ten = test_frame.iloc[:10, :]
    model = OpenAIModel({"model": "gpt-4o"}, is_async=False)

    # Time iterative processing
    start_time_iterative = time.time()
    output_frame_iterative = process_dataframe_iterative(model, test_frame_first_ten)
    end_time_iterative = time.time()
    iterative_duration = end_time_iterative - start_time_iterative

    print(f"Iterative processing time: {iterative_duration:.2f} seconds")

if __name__ == "__main__":
    main()