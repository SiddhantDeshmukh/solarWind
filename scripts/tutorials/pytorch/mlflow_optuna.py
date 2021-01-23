# An integration with MLFlow, PyTorch and Optuna for MNIST

# MLFlow:
# 'Run': this is an execution environment for a piece of ML code that lets
# us log info
# 'Experiment': Tracking component that allows us to group runs following
# specified custom criteria
# 'Parameters': Represent input params for training, like initial LR
# 'Metrics': Used to track progress of training
# 'Artifacts': Can represent any kind of file to save during training

# Optuna:
# 'Objective function': Contains all code of ML task to optimise
# hyperparams for, e.g. traininga and validation loop
# 'Trial': Single call of objective function
# 'Study': Set of trials to be run - best chosen at end of trial

# =========================================================================
# Imports
# =========================================================================
import optuna
import mlflow
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# =========================================================================
# CNN Architecture
# =========================================================================
class Net(nn.Module):
	def __init__(self, dropout=0.0):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv1 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout2d(dropout)
		self.dropout2 = nn.Dropout2d(dropout)
		self.fc1 = nn.Linear(9216, 28)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x: torch.Tensor):
		# Convolutional layers
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)

		# Fully connected layers
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)

		output = F.log_softmax(x, dim=1)
		return output

# =========================================================================
# Training / validation functions
# =========================================================================
def train(model, device, train_loader, optimizer, epoch):
	# Train on mini-batches of MNIST
	model.train()
	train_set_size = len(train_loader.dataset)
	num_batches = len(train_loader)
	train_loss = 0.0

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()

		# Print loss every 10 batches
		if batch_idx % 10 == 0:
			batch_size = len(data)
			print(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{train_set_size} "
						f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}")

	avg_train_loss = train_loss / num_batches
	return avg_train_loss

def validate(model, device, val_loader):
	model.eval()
	val_set_size = len(val_loader.dataset)
	val_loss = 0
	correct = 0

	with torch.no_grad():
		for data, target in val_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			# Sum of batch loss
			val_loss += F.nll_loss(output, target, reduction='sum').item()
			prediction = output.argmax(dim=1, keepdim=True)
			correct += prediction.eq(target.view_as(prediction)).sum().item()

	val_loss /= val_set_size

	print(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{val_set_size} "
          f"({100. * correct / val_set_size:.0f}%)\n")

	return val_loss

# =========================================================================
# Data loading functions
# =========================================================================
def get_mnist_dataloaders(batch_size=8):
	# Load the MNIST train and test datasets and save them to ./data
	mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
															transforms.ToTensor(),
															transforms.Normalize((0.1307,), (0.3081,))
															]))
	train_loader = torch.utils.data.DataLoader(mnist_train,
																						batch_size=batch_size,
																						shuffle=True)
	mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
																							transforms.ToTensor(),
																							transforms.Normalize((0.1307,), (0.3081,))
																					]))
	val_loader = torch.utils.data.DataLoader(mnist_test,
																						batch_size=1000,
																						shuffle=True)

	return train_loader, val_loader

# =========================================================================
# Optuna study & MLFlow run
# =========================================================================
optimizers = {
	"Adam": optim.Adam,
	"Adadelta": optim.Adadelta}

def suggest_hyperparameters(trial):
	# learning rate on log scale
	# dropout ratio in range [0.0, 0.9] (0.1)
	# Optimizer to use as categorical value
	lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
	dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
	optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

	return lr, dropout, optimizer_name

def objective(trial):
	best_val_loss = float('Inf')

	# Start new MLFlow run
	with mlflow.start_run():
		# Get hyparam suggestions created by optuna and log as params w/ mlflow
		lr, dropout, optimizer_name = suggest_hyperparameters(trial)
		mlflow.log_params(trial.params)

		# Use CUDA if GPU available & log device
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		mlflow.log_param("device", device)

		# Initialise network
		model = Net(dropout=dropout).to(device)

		# Pick optimiser based on Optuna's suggestion
		optimizer = optimizers[optimizer_name](model.parameters(), lr=lr)

		scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

		# Get DataLoaders for MNIST train and val set
		train_loader, val_loader = get_mnist_dataloaders(batch_size=8)

		# Network training and validation loop
		for epoch in range(0, 5):
			avg_train_loss = train(model, device, train_loader, optimizer, epoch)
			avg_val_loss = validate(model, device, val_loader)

			if avg_val_loss <= best_val_loss:
				best_val_loss = avg_val_loss
			
			# Log avg train and val set loss metrics for current epoch
			mlflow.log_metric('avg_train_losses', avg_train_loss, step=epoch)
			mlflow.log_metric('avg_val_loss', avg_val_loss, step=epoch)

			scheduler.step()

	return best_val_loss

if __name__ == "__main__":
	# Create Optuna study
	study = optuna.create_study(study_name='pytorch-mlflow-optuna', direction='minimize')
	study.optimize(objective, n_trials=5)

	# Print Optuna study statistics
	print("\n++++++++++++++++++++++++++++++++++\n")
	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))

	print("Best trial:")
	trial = study.best_trial

	print(f" Trial number: {trial.number}")
	print(f" Loss (trial value): {trial.value}")

	print(" Params:")
	for key, value in trial.params.items():
		print(f"{key}: {value}")
