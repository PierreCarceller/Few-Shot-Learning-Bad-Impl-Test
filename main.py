#!/usr/bin/python3
import random
from FewShotLearningService import FewShotLearningService
from FewShotLearningService import FewShotLearningDataLoader
from FewShotLearningService import Dataset
from pathlib import Path

path = Path("insert path here")
path2 = Path("insert path here")

try:
    data = FewShotLearningDataLoader(data_path=Path("data"), support_set_path=Path("support_set"))
    data = data.data
except Exception as e:
    exit(0)
    
service = FewShotLearningService(model="facebook/deit-base-distilled-patch16-224-in21k")
support_set = service.support_set(images=data.support_set_imgs, classes=data.classes)

query_set = random.sample(data.images, 5)

for image in query_set:
    print(service.classify(image, support_set))
    
print("Done")

