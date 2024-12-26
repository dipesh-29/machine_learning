# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected Network
# Mnist dataset size 28x28 
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Pytorch_ANN():
    # Initialise Parameters
    def __init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Hyperparameters
        self.input_size = 784
        self.num_classes = 10
        self.learning_rate = 0.001
        self.batch_size = 64
        self.num_epochs = 3
        # Initialize Network
        self.model = NN(input_size=self.input_size, num_classes=self.num_classes).to(self.device)
        # Loass and Optmiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    # Load Data
    def load_data(self):
        train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    # Train Network
    def train_model(self, train_loader):
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                
                data = data.reshape(data.shape[0], -1)

                # forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # backward
                self.optimiser.zero_grad()
                loss.backward()

                # Gredient descent or adam step
                self.optimiser.step()
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')


    # Check Accuracy on training and test to see model performance
    def check_accuracy(self, loader):
        num_correct = 0 
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x = x.reshape(x.shape[0], -1)

                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            print(f'Got {num_correct}/{num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}')


if __name__=="__main__":
    ann_obj = Pytorch_ANN()
    train_loader, test_loader = ann_obj.load_data()
    ann_obj.train_model(train_loader)
    ann_obj.check_accuracy(train_loader)
    ann_obj.check_accuracy(test_loader)