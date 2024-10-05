import pandas as pd
import numpy as np
import openai
import tenacity
from PIL.Image import Image
import base64
from abc import ABC, abstractmethod
import re
import asyncio
import time
from tqdm.asyncio import tqdm
from typing import Dict, List

# Open the file and read the line
with open('/home/joseph/keys/openai-key.txt', 'r') as file:
    openaikey = file.readline().strip()

class UnanswerableError(Exception):
    """An exception indicating the prompt was too long for the model."""

def parse_answers(text):
    match = re.search(r'<ANS>\s*([A-Z])\s*\.?\s*</?ANS>', text)
    if match:
        # Return the single capital letter
        return match.group(1)
    else:
        # Return None if the expected format isn't found
        return None

class OpenAIModel(ABC):
    def __init__(self, model_kwargs: dict, is_async=False, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs = model_kwargs.copy()
        self.model_kwargs.setdefault("model", "gpt-4o")
        self.detail = "high"
        if is_async:
            self.client = openai.AsyncOpenAI(api_key=openaikey)
        else:
            self.client = openai.OpenAI(api_key=openaikey)
        
    def generate_text_url(self, text):
        return {"type": "text", "text": text}
        
    def generate_image_url(self, image_path, detail="high"):
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64, {encode_image(image_path)}",
                "detail": detail,
            },
        }

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),  # Increase the number of retries
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Increase wait time between retries
        retry=tenacity.retry_if_exception(
            lambda exc: not isinstance(exc, UnanswerableError)
        ),
    )
    async def get_completion_async(self, text_prompt: str, test_cxr: str, demo_cxr: list[str]) -> str:
        image_paths = demo_cxr + [test_cxr]
        prompt = text_prompt.split("<<IMG>>")
#         print(prompt)
#         print("Len prompt: ", len(prompt))
#         print("Len image paths: ", len(image_paths))
        assert len(prompt) == len(image_paths) + 1
        if prompt[0] != "":
            messages = [self.generate_text_url(prompt[0])]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            messages.append(
                self.generate_image_url(image_paths[idx - 1], detail=self.detail)
            )
            if prompt[idx].strip() != "":
                messages.append(self.generate_text_url(prompt[idx]))
        
        try:
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": messages}],
                **self.model_kwargs,
            )
        except openai.BadRequestError as e:
            if "PromptTooLongError" in e.message:
                raise UnanswerableError(e.message) from e
            raise

        return response.choices[0].message.content
    
    def get_completion(self, text_prompt: str, test_cxr: str, demo_cxr: list[str] | None) -> str:
        if demo_cxr:
            image_paths = demo_cxr
            image_paths.append(test_cxr)
            prompt = text_prompt.split("<<IMG>>")
            assert len(prompt) == len(image_paths) + 1
            if prompt[0] != "":
                messages = [self.generate_text_url(prompt[0])]
            else:
                messages = []
            for idx in range(1, len(prompt)):
                messages.append(
                    self.generate_image_url(image_paths[idx - 1], detail=self.detail)
                )
                if prompt[idx].strip() != "":
                    messages.append(self.generate_text_url(prompt[idx]))
        else:
            messages = []
            messages.append(self.generate_image_url(test_cxr, detail=self.detail))
            messages.append(self.generate_text_url(text_prompt.split("<<IMG>>")[1]))
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": messages}],
                **self.model_kwargs,
            )
        except openai.BadRequestError as e:
            if "PromptTooLongError" in e.message:
                raise UnanswerableError(e.message) from e
            raise

        return response.choices[0].message.content
    
async def process_row(model, row, demo_prompt, demo_paths, semaphore):
    async with semaphore:
        ground_truth = row.malignant
        fst = '12' if row.skin_tone == 12 else '56'
        path = image_path + row.DDI_file

#         formatted_columns = ", ".join([f"{chr(65+i)}. {col}" for i, col in enumerate(row.index[5:19])])
        prompt = f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: Which of the following descriptors best describes the image?
Choices: A. Benign, B. Malignant
"""
    
        prompt = demo_prompt + prompt

        response = await model.get_completion_async(prompt, path, demo_paths)
        ans = parse_answers(response)

        return {
            "response": response,
            "parsed_answer": ans,
            "path": path,
            "ground_truth": ground_truth,
            "race": fst
        }

async def process_dataframe_async(model, test_frame, demo_prompt, demo_paths, semaphore):
    tasks = []
    for _, row in test_frame.iterrows():
        task = process_row(model, row, demo_prompt, demo_paths, semaphore)
        tasks.append(task)
        
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing rows"):
        result = await task
        results.append(result)

    result_df = pd.DataFrame(results)
    return result_df

def create_demo_frame(demo_frame: pd.DataFrame, sample_sizes: Dict[str, int], random_state: int = 42) -> pd.DataFrame:
    frames = []
    for skin_tone in [56, 12]:
        for malignant in [True, False]:
            subset = demo_frame[(demo_frame.skin_tone == skin_tone) & (demo_frame.malignant == malignant)]
            key = f"fst{skin_tone}_{'mal' if malignant else 'ben'}"
            frames.append(subset.sample(sample_sizes[key], random_state=random_state))
    
    return pd.concat(frames).sample(frac=1, random_state=random_state)

def create_demo_prompt(final_demo_frame: pd.DataFrame, image_path: str) -> tuple[str, List[str]]:
    demo_prompt = """Please respond with the following format for each question, in the form of a single capital letter specifying which label best describes the image surrounded by beginning <ANS> and end </ANS> brackets:
        ---BEGIN FORMAT TEMPLATE FOR QUESTION---
        <ANS> The single capital letter best describing the image here </ANS>
        ---END FORMAT TEMPLATE FOR QUESTION---

        Do not deviate from the above format, because it will be automatically parsed with a regex."""
    
    demo_paths = []
    for _, row in final_demo_frame.iterrows():
        ans_choice = "B" if row.malignant else "A"
        demo_prompt += f"""<<IMG>>Given the image above, answer the following question using the specified format. 
        Question: Which of the following descriptors best describes the image?
        Choices: A. Benign, B. Malignant
        Answer: <ANS> {ans_choice} </ANS>
        
        """
        demo_paths.append(image_path + row.DDI_file)
    
    return demo_prompt, demo_paths

async def run_experiment(demo_frame_path: str, test_frame_path: str, image_path: str, sample_sizes: Dict[str, int], output_file: str):
    demo_frame = pd.read_csv(demo_frame_path, index_col=0)
    final_demo_frame = create_demo_frame(demo_frame, sample_sizes)
    demo_prompt, demo_paths = create_demo_prompt(final_demo_frame, image_path)
    
    test_frame = pd.read_csv(test_frame_path, index_col=0)
    async_model = OpenAIModel({"model": "gpt-4o"}, is_async=True)
    
    semaphore = asyncio.Semaphore(5)

    start_time_async = time.time()
    output_frame_async = await process_dataframe_async(async_model, test_frame, demo_prompt, demo_paths, semaphore)
    output_frame_async.to_csv(output_file)
    end_time_async = time.time()
    async_duration = end_time_async - start_time_async

    print(f"Async processing time: {async_duration:.2f} seconds")

async def run_multiple_experiments(experiments: List[Dict]):
    for exp in experiments:
        await run_experiment(**exp)

if __name__ == "__main__":
    global image_path
    image_path = "path/to/images"
    experiments = [
        {
            "demo_frame_path": "path/to/demo_frame.csv",
            "test_frame_path": "path/to/test_frame.csv",
            "image_path": "path/to/images/",
            "sample_sizes": {
                "fst56_mal": 5,
                "fst56_ben": 15,
                "fst12_mal": 5,
                "fst12_ben": 15
            },
            "output_file": "output_experiment1.csv"
        },
        # Add more experiments here with different sample sizes or paths
    ]
    
    asyncio.run(run_multiple_experiments(experiments))
