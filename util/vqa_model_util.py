from PIL import Image
import torch
import random
import math
import requests
from io import BytesIO
from util.InternVL2_util import load_image, split_model
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(1)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Base handler

class BaseModelWrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self, model_name: str):
        raise Exception("Not Implemented")
    
    def ask(self, input_string: str, img_dir: str):
        raise Exception("Not Implemented")
        
    def load_img(self, img_dir):
        if img_dir.startswith('http://') or img_dir.startswith('https://'):
            response = requests.get(img_dir)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(img_dir).convert('RGB')

        return image
    
    def __repr__(self):
        return f"<Model Handler for: {self.model_name}>"


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for InternVL

class InternVL2_Wrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('InternVL2-4B')

        self.load_model()

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer
        path = f'OpenGVLab/InternVL2-4B' # 'OpenGVLab/InternVL-Chat-V1-5' # OpenGVLab/InternVL2-4B
        device_map = split_model('InternVL2-4B')
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    

    def ask(self, q_str, img_list):
        # multi-image multi-round conversation, combined images
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
        pixel_values = torch.cat(pixel_value_list, dim=0)

        text_prompt = (f'Answer the question:\n{q_str}\nPlease make the answer as concise as possible')
        
        question = f'<image>\n{text_prompt}'
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)

        return response

    def ask_sep(self, q_str, img_list):
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
        pixel_values = torch.cat(pixel_value_list, dim=0)
        num_patches_list = [pixel_values.size(0) for pixel_values in pixel_value_list]
    
        prompt_img_prefix = '\n'.join([f'Image-{x}: <image>' for x in range(1, len(img_list)+1)])

        text_prompt = (f'Answer the question:\n{q_str}\nPlease make the answer as concise as possible')
        
        question = prompt_img_prefix + f'\n{text_prompt}'
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                           num_patches_list=num_patches_list,
                                           history=None, return_history=True)
        return response


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for InternVL2-Lora

class LoraInternVL_Wrapper(BaseModelWrapper):
    def __init__(self, lora_dir=None):
        super().__init__('LoraInternVL2')
        self.load_model(lora_dir=lora_dir)

    def load_model(self, lora_dir):
        path = f'OpenGVLab/InternVL2-4B' # 'OpenGVLab/InternVL-Chat-V1-5' # OpenGVLab/InternVL2-8B
        device_map = split_model('InternVL2-4B')
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        if lora_dir is not None:
            self.model.load_adapter(lora_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


    def ask(self, q_str, img_list):

        system_prompt = ('Please analyze the provided image of a page in a document or slide '
                         'and answer the following question based on the visual and textual information present in the image. '
                         'Keep your answer concise and output only the answer without any explanation.\n')
        
        # multi-image multi-round conversation, combined images
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
        pixel_values = torch.cat(pixel_value_list, dim=0)

        text_prompt = system_prompt + f'Answer the question:\n{q_str}\nPlease make the answer as concise as possible.'
        
        question = f'<image>\n{text_prompt}'
        response, history = self.model.chat(self.tokenizer, pixel_values, question, generation_config,
                                       history=None, return_history=True)
        return response



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Handler for Phi-3-Vision

class Phi3V_Wrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__('Phi-3v')

        self.load_model()

    def load_model(self):
        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor 
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    def ask(self, question_string: str, img_dir_list: str):

        if len(img_dir_list) == 0:
            return ''
            
        img_dir = img_dir_list[0]
        
        # prepare prompt
        prompt = (f'A chat between a curious human and an artificial intelligence assistant. '
              f'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
              f'USER: {question_string}\nPlease make the answer as concise as possible. ASSISTANT:')

        # setup message
        # we do zero-shot test, so ignore demonstrations
        messages = [
        # {"role": "user", "content": f"<|image_1|>\n{demo_q}"}, 
        # {"role": "assistant", "content": f"{demo_a}"}, 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"} 
        ]

        # load image from dir
        image = self.load_img(img_dir)
        
        prompt_in = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt_in, [image], return_tensors="pt").to("cuda")
        
        generation_args = { 
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        model_answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        
        return model_answer


