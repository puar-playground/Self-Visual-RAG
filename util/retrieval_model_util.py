from PIL import Image
import os
import torch
import requests
from io import BytesIO
from safetensors.torch import load_file
from typing import List, Tuple, Union
os.system('clear')
from abc import ABC, abstractmethod

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return image

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BaseRetriever(ABC):
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColPhi, ColPhiProcessor

class ColPhiRetriever(BaseRetriever):
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColInternvl2_4b, ColInternProcessor

class ColInternVL2Retriever(BaseRetriever):
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

        self.processor = ColInternProcessor('OpenGVLab/InternVL2-4B')

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

                img = load_image(img_dir)
        
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
