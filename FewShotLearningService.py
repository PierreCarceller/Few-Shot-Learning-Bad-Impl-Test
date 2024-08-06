#!/usr/bin/python3
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2
import os



class Dataset:
    
    def __init__(self, data_path: Path, support_set_path: Path, support_set_imgs, classes, images):
        self.data_path = data_path
        self.support_set_path = support_set_path
        self.support_set_imgs = support_set_imgs
        self.classes = classes
        self.images = images


class FewShotLearningDataLoader:
    
    def __init__(self, data_path: Path, support_set_path: Path):
        self.data_path = data_path
        self.support_set_path = support_set_path
        
    @property
    def data(self) -> Dataset:
        images = self.support_set_path.glob("*.jpg")
        classes = sorted(np.unique([image.parent.name for image in images]).tolist())
        
        support_set_imgs = []
        for c in classes:
            f = list(filter(lambda x: x.parent.name == c, images))
            support_set_imgs += f[:5]
            
        images = self.data_path.glob("*.jpg")
        
        return Dataset(self.data_path, self.support_set_path, support_set_imgs, classes, images)
          

class FewShotLearningService:
    
    def __init__(self, model: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.image_processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def support_set(self, images: List[str], classes: List[str]) -> Dict[str, torch.Tensor]:
        
        count = 0
        k = 5
        d = dict()
        for current_class in classes:
            for _ in range(k):
                
                if d.get(current_class) is None:
                    d[current_class] = list()
                
                current_image = Image.open(images[count])
                inputs = self.image_processor(current_image, return_tensors="pt")
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
                
                count += 1
                
                d[current_class].append(embeddings)
                
            d[current_class] = torch.stack(d[current_class]).mean(dim=0)
        
        return d
    
    def classify(self, query_image: str, support_set: Dict[str, torch.Tensor]) -> str:
        
        def euclidean_distance(embeddings1, embeddings2):
            return torch.nn.functional.pairwise_distance(embeddings1, embeddings2, p=2).item()
        
        def cosine_similarity(embeddings1, embeddings2):
            return torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
        
        query_image = Image.open(query_image)
        inputs = self.image_processor(query_image, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output
        
        max_sim = -1
        predicted_class = None
        sims = []
        for current_class, support_embedding in support_set.items():
            sim = cosine_similarity(embeddings, support_embedding)
            if sim > max_sim:
                max_sim = sim
                predicted_class = current_class
                
        return predicted_class
    
