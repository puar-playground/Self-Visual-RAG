from PIL import Image
import os
import torch
import requests
from io import BytesIO
from typing import List, Tuple, Union
os.system('clear')

from peft import PeftModel
from abc import ABC, abstractmethod
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM 
from transformers import AutoModel, AutoTokenizer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BaseModel(ABC):
    """
    A base class for retriever models that process both text and images 
    and compute similarities for retrieval tasks.
    """

    @abstractmethod
    def retrieve(self, query, image_list, top_k=5):
        """
        Retrieves the most relevant images from a list based on the given query.

        Args:
            query (str): The input query text.
            image_list (list of str): List of image file paths.
            top_k (int): Number of top relevant images to return (default: 5).

        Returns:
            list of tuples: Sorted list of (image_path, similarity_score), descending order.
        """
        pass

    @abstractmethod
    def compute_similarity(self, item1, item2):
        """
        Computes similarity between two items (text or image).
        
        Args:
            item1: First item (can be text or image).
            item2: Second item (can be text or image).
        
        Returns:
            A similarity score (float).
        """
        pass

    @abstractmethod
    def process_text(self, text):
        """
        Processes text input (e.g., tokenization, embedding).
        
        Args:
            text (str): List of input text strings to process.
        
        Returns:
            Processed text representation.
        """
        pass

    @abstractmethod
    def process_image(self, image_list):
        """
        Processes image input (e.g., feature extraction, encoding).
        
        Args:
            image_list: List of input image directories to process.
        
        Returns:
            Processed image representation.
        """
        pass

    @staticmethod
    def load_img(img_dir):
        if img_dir.startswith('http://') or img_dir.startswith('https://'):
            response = requests.get(img_dir)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(img_dir)

        return image
    
    @abstractmethod
    def ask(self, question_string: str, img_dir_list: str):
        """
        Generate answer for question on the given image.
        """
        pass


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColPhi, ColPhiProcessor

class SVRAG_Phi(BaseModel):
    """Retriever class using ColPhi for multimodal retrieval."""

    def __init__(self, model_name="puar-playground/Col-Phi-3-V", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the ColPhi model.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install transformers==4.47.1')
        self.multimodel = True
        self.device = device

        self.model = ColPhi.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor = ColPhiProcessor.from_pretrained(model_name)

        self.lm_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True, torch_dtype="auto")
        self.lm_model.model.load_state_dict(self.model.model.state_dict(), strict=False)
        self.lm_model.to(device)

        self.lm_processor = AutoProcessor.from_pretrained("microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True)

    @staticmethod
    def pad_and_cat_tensors(tensor_list):
        # Find the maximum length of the second dimension (x_i) across all tensors
        max_x = max(tensor.size(1) for tensor in tensor_list)
        
        # Pad tensors to have the same size in the second dimension
        padded_tensors = []
        for tensor in tensor_list:
            padding_size = max_x - tensor.size(1)
            # Pad with zeros on the right in the second dimension
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
            padded_tensors.append(padded_tensor)
        
        # Concatenate the padded tensors along the first dimension
        result_tensor = torch.cat(padded_tensors, dim=0)
        
        return result_tensor
    
    def process_text(self, query_list: List[str], batch_size: int = 2):
        """
        Processes a list of text queries into embeddings using ColPhi in batches.

        Args:
            query_list (List[str]): List of query texts.
            batch_size (int): Number of queries processed per batch.

        Returns:
            torch.Tensor: Concatenated embeddings for all queries.
        """
        all_embeddings = []

        for i in range(0, len(query_list), batch_size):
            batch_queries = query_list[i : i + batch_size]

            # Convert queries to model-compatible format
            batch_inputs = self.processor.process_queries(batch_queries).to(self.model.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)

            all_embeddings.append(batch_embeddings.to("cpu"))
        
        # Concatenate all processed batches into a single tensor
        all_embeddings = self.pad_and_cat_tensors(all_embeddings)

        # Concatenate all batch outputs into a single tensor
        return all_embeddings

    def process_image(self, image_dir_list: List[str]):
        """Processes images into embeddings using ColPhi."""
        def process_images_in_batches(processor, img_dir_list, model, batch_size=1):
            def load_image(image_file):
                if image_file.startswith('http://') or image_file.startswith('https://'):
                    response = requests.get(image_file)
                    image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image_file)
                return image
            
            all_embeddings = []
            
            # Split img_dir_list into batches
            for i in range(0, len(img_dir_list), batch_size):
                batch_img_dirs = img_dir_list[i:i + batch_size]
                image_list = [load_image(img_dir) for img_dir in batch_img_dirs]
        
                # Process the batch of images
                batch_features = processor.process_images(image_list)
                
                # Extract the tensor from the BatchFeature object
                batch_images = {k: v.to(model.device) for k, v in batch_features.items()}
        
                # Assuming the model expects a specific input (e.g., 'pixel_values')
                embeddings = model(**batch_images)
                
                # Move embeddings to CPU and append to the list
                embeddings = embeddings.to("cpu")
                all_embeddings.append(embeddings)

            # Concatenate all processed batches into a single tensor
            all_embeddings = torch.cat(all_embeddings, dim=0)
            return all_embeddings
        
        # Forward pass
        with torch.no_grad():
            # image_embeddings = model(**batch_images)
            image_embeddings = process_images_in_batches(self.processor, image_dir_list, self.model)
                    
        return image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """ Computes cosine similarity between text and image embeddings. """
        scores = self.processor.score_multi_vector(text_embeddings, image_embeddings)
        return scores

    def retrieve(self, query_list: str, image_list: List[str]):
        
        with torch.no_grad():
            text_embeddings = self.process_text(query_list)
            image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(text_embeddings, image_embeddings)
        values, top_indices = torch.tensor(similarity_score).sort(descending=True)

        return values, top_indices
    
    def disable_lora_if_present(self):
        """
        Detects if the model has a LoRA adapter and disables it.
        """
        if isinstance(self.model, PeftModel):  # Check if it's a PeftModel (LoRA)
            self.lm_model.disable_adapter()
            print("LoRA adapter detected and disabled.")

    def ask(self, question_string: str, img_dir_list: str):
        
        self.disable_lora_if_present()

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
        image = self.load_img(img_dir).convert('RGB')
        
        prompt_in = self.lm_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.lm_processor(prompt_in, [image], return_tensors="pt").to("cuda")
        
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.lm_model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        model_answer = self.lm_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        
        return model_answer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColInternvl2_4b, ColInternProcessor
from util.InternVL2_util import load_image
from colpali_engine.models.InternVL2.model_4b_util.modeling_internvl_chat import InternVLChatModel_4B, InternVL2PreTrainedModel_4B

class SVRAG_InternVL2(BaseModel):
    """Retriever class using ColInternVL2 for multimodal retrieval."""

    def __init__(self, model_name="puar-playground/Col-InternVL2-4B", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the ColInternVL2 model.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install transformers==4.47.1')
        self.multimodel = True
        self.device = device

        self.model = ColInternvl2_4b.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device).eval()
        
        self.lm_model = self.model.model

        self.processor = ColInternProcessor('OpenGVLab/InternVL2-4B')
        self.tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL2-4B', trust_remote_code=True, use_fast=False)

    def process_text(self, query_list: List[str], batch_size: int = 4):
        """
        Processes a list of text queries into embeddings using ColPhi in batches.

        Args:
            query_list (List[str]): List of query texts.
            batch_size (int): Number of queries processed per batch.

        Returns:
            torch.Tensor: Concatenated embeddings for all queries.
        """
        all_embeddings = []

        for i in range(0, len(query_list), batch_size):
            batch_queries = query_list[i : i + batch_size]

            # Convert queries to model-compatible format
            batch_inputs = self.processor.process_queries(batch_queries).to(self.model.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)

            all_embeddings.append(batch_embeddings.to("cpu"))
        
        # Concatenate all batch outputs into a single tensor
        all_embeddings = self.pad_and_cat_tensors(all_embeddings)

        return all_embeddings
    
    @staticmethod
    def pad_and_cat_tensors(tensor_list):
        # Find the maximum length of the second dimension (x_i) across all tensors
        max_x = max(tensor.size(1) for tensor in tensor_list)
        
        # Pad tensors to have the same size in the second dimension
        padded_tensors = []
        for tensor in tensor_list:
            padding_size = max_x - tensor.size(1)
            # Pad with zeros on the right in the second dimension
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
            padded_tensors.append(padded_tensor)
        
        # Concatenate the padded tensors along the first dimension
        result_tensor = torch.cat(padded_tensors, dim=0)
        
        return result_tensor

    def process_image(self, image_dir_list: List[str]):
        """Processes images into embeddings using ColInternVL2."""
        def process_images_in_batches(processor, img_dir_list, model, batch_size=2):
            all_embeddings = []
            
            # Split img_dir_list into batches
            for img_dir in img_dir_list:

                img = self.load_img(img_dir)
        
                # Process the batch of images
                batch_features = processor.process_images(img)
                
                # Extract the tensor from the BatchFeature object
                batch_images = {k: v.to(model.device) for k, v in batch_features.items()}
        
                # Assuming the model expects a specific input (e.g., 'pixel_values')
                embeddings = model(**batch_images)
                
                # Move embeddings to CPU and append to the list
                embeddings = embeddings.to("cpu")
                all_embeddings.append(embeddings)

            # Concatenate all processed batches into a single tensor
            all_embeddings = self.pad_and_cat_tensors(all_embeddings)
            return all_embeddings
        
        # Forward pass
        with torch.no_grad():
            # image_embeddings = model(**batch_images)
            image_embeddings = process_images_in_batches(self.processor, image_dir_list, self.model)
                    
        return image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """ Computes cosine similarity between text and image embeddings. """
        scores = self.processor.score_multi_vector(text_embeddings, image_embeddings)
        return scores

    def retrieve(self, query_list: str, image_list: List[str]):

        text_embeddings = self.process_text(query_list)
        image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(text_embeddings, image_embeddings)
        values, top_indices = torch.tensor(similarity_score).sort(descending=True)

        return values, top_indices

    def disable_lora_if_present(self):
        """
        Detect and disable LoRA layers inside self.lm_model by setting weights to zero.
        """
        if not hasattr(self, "lm_model"):
            print("❌ self.lm_model does not exist.")
            return

        for name, module in self.lm_model.named_modules():

            # If module supports `disable_adapter()`, call it
            if hasattr(module, "disable_adapter"):
                module.disable_adapter()
                # print(f"✅ Adapter disabled in {name}")

            elif hasattr(module, "lora_A") and hasattr(module, "lora_B"):  # LoRA layers exist
                module.lora_A.default.weight.data.zero_()  # Disable LoRA weights
                module.lora_B.default.weight.data.zero_()
                # print(f"✅ LoRA disabled in {name}")

    def ask(self, q_str, img_list):

        # multi-image multi-round conversation, combined images
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
        pixel_values = torch.cat(pixel_value_list, dim=0)

        text_prompt = (f'Answer the question:\n{q_str}\nPlease make the answer as concise as possible')
        
        question = f'<image>\n{text_prompt}'
        with torch.no_grad():
            response, history = self.lm_model.chat(self.tokenizer, pixel_values, question, generation_config,
                                           history=None, return_history=True)
            

        return response
