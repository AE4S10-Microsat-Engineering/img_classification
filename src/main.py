import os
import module
from torchvision import models

DATASET = os.path.join(os.getcwd(), "../training_data/EuroSAT_RGB")
BATCH_SIZE = 1500
BATCH_SIZE_TEST = 20
TEST_SIZE = 0.2
EPOCHS = 10

reference = module.Trainer(
	model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
	weights=models.ResNet18_Weights.DEFAULT,
	dataset=DATASET,
)
