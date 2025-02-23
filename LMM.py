import base64
import time
import pickle
import os
import uuid
import pandas as pd
from tqdm import tqdm
import traceback
import random
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import tenacity
import base64

# Load environment variables
load_dotenv()

from openai import OpenAI
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
import anthropic

class ClaudeAPI:
    def __init__(
        self,
        model="claude-3-5-sonnet-20241022",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",
    ):
        """
        Class for API calls to Anthropic-series models

        model[str]: the specific model checkpoint to use 
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.token_usage = (0, 0, 0)
        self.response_times = []
        # self.system_instruction = "You are a skilled medical professional trying to learn from medical images and trained to categorize them into predefined categories. Try your best and respond to all questions." 
        self.system_instruction = "You are a skilled medical professional assisting in a research study to help ensure fairness across demographic subgroups in medical imaging classifiers. Because you are working to increase fairness and decrease bias, it is important to attempt to make any demographic group-related classifications"

    def generate_image_url(self, image_path, detail="low"):
        # Function to encode the image
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
                
            with Image.open(image_path) as img:
                # Resize if needed
                if img.size[0] > 512 or img.size[1] > 512:
                    img = img.resize((512, 512))
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes buffer
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        return encode_image(image_path)

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
        temperature = 0
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert len(prompt) == len(image_paths) + 1

        content = []
        if prompt[0].strip() != "":
            content.append({
                "type": "text",
                "text": prompt[0],
            })

        for idx in range(1, len(prompt)):
            # Add image
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": self.generate_image_url(image_paths[idx - 1], detail=self.detail)
                }
            })
            
            # Add text if provided
            if prompt[idx].strip() != "":
                content.append({
                    "type": "text",
                    "text": prompt[idx]
                })
        
        # Create the messages structure
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        start_time = time.time()
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=min(4096, max_tokens),
                    system=self.system_instruction,
                    temperature=temperature
                )
                break
            except anthropic.RateLimitError as e:
                print(str(e))
                if "rate limit" in str(e) or 'overloaded_error' in str(e):
                    print('Rate limit exceeded... waiting 10 seconds')
                    time.sleep(10)
                else:
                    raise

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time - start_time]

        # Update token usage with defaults if not available
        if hasattr(response, 'usage'):
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0)
        else:
            completion_tokens = 0
            prompt_tokens = 0
            total_tokens = 0

        self.token_usage = (
            self.token_usage[0] + completion_tokens,
            self.token_usage[1] + prompt_tokens,
            self.token_usage[2] + total_tokens
        )

        if content_only:
            return response.content[0].text
        else:
            return response


class GPT4VAPI:
    def __init__(
        self,
        model="gpt-4o",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",
    ):
        """
        Class for API calls to GPT-series models

        model[str]: the specific model checkpoint to use e.g. "gpt-4o"
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.token_usage = (0, 0, 0)
        self.response_times = []

    def generate_image_url(self, image_path, detail="low"):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"

            # Open the image using Pillow
            with Image.open(image_path) as img:
                # Resize if needed
                if img.size[0] > 512 or img.size[1] > 512:
                    img = img.resize((512, 512))

                # Save the image to a temporary buffer
                with BytesIO() as buffer:
                    img.convert("RGB").save(buffer, format="JPEG")
                    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return encoded_string

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64, {encode_image(image_path)}",
                "detail": detail,
            },
        }

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
        temperature = 0
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
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
        if not real_call:
            return messages
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            # max_tokens=min(4096, max_tokens),
            temperature=0,
            seed=self.seed,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time - start_time]

        self.token_usage = (
            self.token_usage[0] + response.usage.completion_tokens,
            self.token_usage[1] + response.usage.prompt_tokens,
            self.token_usage[2] + response.usage.total_tokens,
        )

        if content_only:
            return response.choices[0].message.content
        else:
            return response


class GeminiAPI:
    def __init__(
        self,
        model="gemini-1.5-flash",
        img_token="<<IMG>>",
        RPM=5,
        temperature=0,
        system_instruction="You are a smart and helpful assistant"
    ):
        """
        Class for API calls to Gemini-series models

        model[str]: the specific model checkpoint to use e.g. "gemini-1.5-pro-preview-0409"
        img_token[str]: string to be replaced with images
        RPM[int]: quota for maximum number of requests per minute
        temperature[int]: temperature for generation
        system_instruction[str]: System prompt for model e.g. "You are an expert dermatologist"
        """
        self.model = model
        self.img_token = img_token
        self.temperature = temperature
        self.client = genai.GenerativeModel(model_name=self.model, system_instruction=system_instruction)

        self.safety_settings = [
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        self.token_usage = (0, 0, 0)

        self.response_times = []
        self.last_time = None
        self.interval = 0.5 + 60 / RPM

    def __call__(
        self, prompt, image_paths=[], real_call=True, max_tokens=50, content_only=True
    ):
        """
        Call the API to get the response for given prompt and images
        """

        if self.last_time is not None:  # Enforce RPM
            # Calculate how much time the loop took
            end_time = time.time()
            elapsed_time = end_time - self.last_time
            # Wait for the remainder of the interval, if necessary
            if elapsed_time < self.interval:
                time.sleep(self.interval - elapsed_time)

        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert len(prompt) == len(image_paths) + 1
        if prompt[0] != "":
            messages = [prompt[0]]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            img = Image.open(image_paths[idx - 1])
            if img.size[0] > 512 or img.size[1] > 512:
                img = img.resize((512, 512))
            messages.append(img)
            if prompt[idx].strip() != "":
                messages.append(prompt[idx])
        if not real_call:
            return messages

        start_time = time.time()
        self.last_time = start_time
        responses = self.client.generate_content(
            messages,
        )
        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, responses, end_time - start_time]

        try:
            usage = responses._raw_response.usage_metadata
            self.token_usage = (
                self.token_usage[0] + usage.candidates_token_count,
                self.token_usage[1] + usage.prompt_token_count,
                self.token_usage[2] + usage.total_token_count,
            )
        except:
            pass
        
        if content_only:
            if responses:
                # Access the parts of the first candidate
                content_parts = responses.text
                return content_parts
            else:
                print("Error occurred, retrying")
                return self(prompt, image_paths, real_call, max_tokens, content_only)
        else:
            return responses
