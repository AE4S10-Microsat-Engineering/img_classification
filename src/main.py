import os
import module
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

DATASET = os.path.join(os.getcwd(), "../training_data/EuroSAT_RGB")
BATCH_SIZE = 1500
BATCH_SIZE_TEST = 20
TEST_SIZE = 0.2
EPOCHS = 10

reference = module.Trainer(
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
	weights = models.ResNet18_Weights.DEFAULT,
	dataset = DATASET,
	
)