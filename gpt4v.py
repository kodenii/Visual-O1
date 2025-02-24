
import base64
from io import BytesIO
from openai import AzureOpenAI
import time
from PIL import Image


meta_prompt = '''You are an helpful assistant.'''    


def encode_image_from_pil(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class Content():
    def __init__(self, system_prompt):
        if system_prompt is None:
            self.system_prompt = "You are an helpful assistant."
        else:
            self.system_prompt = system_prompt
        system_message = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        self.history = []
        self.history.append(system_message)
    
    def add_message(self, text: str, image: list):
        last_message = self.history[-1]
        last_message_role = last_message["role"]
        if last_message_role == "system":
            role = "user"
        elif last_message_role == "user":
            role = "assistant"
        else:
            role = "user"
        text_content = [{"type": "text", "text": text}]
        message = {"role": role, "content": text_content}
        if image is not None:
            
            base64_image = encode_image_from_pil(image)
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            message["content"].append(image_message)
        
        self.history.append(message)
            

class Agent():
    
    def __init__(self, model, api_key, api_version, azure_endpoint, system_prompt=None):
        self.model = model
        self.api_key = api_key
        self.content = Content(system_prompt)
        self.client = AzureOpenAI(
            api_key = api_key,
            api_version = api_version,
            azure_endpoint = azure_endpoint
        )
    def chat(self, text, image=None):
        self.content.add_message(text, image)
        messages = self.content.history
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model, # model = "deployment_name".
                    messages=messages,
                    max_tokens=800,
                    timeout=30
                )
                break
            except Exception as e:
                print("wait 10s & retry")
                time.sleep(10)
        res = response.choices[0].message.content
        self.content.add_message(res, None)
        return res

