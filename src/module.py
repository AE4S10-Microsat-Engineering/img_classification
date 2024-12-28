import torch.nn as nn

class Trainer():
	def __init__(self, model, weights, dataset):
		self.model = model
		# Set the amount of classes
		self.model.fc = nn.Linear(self.model.fc.in_features, dataset.num_classes)
		self.preprocess = weights.transforms()

		self.train_dataset = dataset
		self.validate_dataset = dataset
		self.test_dataset = dataset

		self.loss = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

		# Check if GPU is available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# Transfer model to device
		self.model = self.model.to(self.device)

	def train(self):
		return