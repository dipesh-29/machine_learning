# Import 
import pandas as pd
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Sentiment_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Sentiment_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0,c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTM_Helper():
    def __init__(self):
        # Set Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # Parameters
        self.max_length = 0
        self.input_size = self.max_length  # Number of features
        self.hidden_size = 70  # Number of hidden units
        self.output_size = 2  # Number of output classes
        self.num_layers = 1  # Number of RNN layers
        self.batch_size = 5 # Batch Size
        self.seq_len = 1 # Sequence length
        self.num_epochs = 5
        self.word_to_idx = {}

    # Preprocess dataset
    def preprocess_text(self, text):
        return text.lower().split()

    def encode_phrase(self, phrase):
        return [self.word_to_idx[word] for word in phrase]

    def pad_sequence(self, seq, max_length):
        return seq + [0] * (max_length - len(seq))
        
    def load_data(self):
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        # Load dataset
        url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
        df = pd.read_csv(url, delimiter='\t', header=None, names=['label', 'text'])
        print(df.head())
        df['text'] = df['text'].apply(self.preprocess_text)
        df = df[['text', 'label']]
        # Encode labels
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        # Split dataset
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        
        # Vocabulary and indexing
        vocab = set([word for phrase in df['text'] for word in phrase])
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}

        train_data['text'] = train_data['text'].apply(self.encode_phrase)
        test_data['text'] = test_data['text'].apply(self.encode_phrase)
        
        # Padding sequences
        self.max_length = max(df['text'].apply(len))
        self.input_size = self.max_length
        
        train_data['text'] = train_data['text'].apply(lambda x: self.pad_sequence(x, self.max_length))
        test_data['text'] = test_data['text'].apply(lambda x: self.pad_sequence(x, self.max_length))
        return train_data, test_data

    def data_loader(self, train_data, test_data):
        # Create a TensorDataset
        data_train = torch.tensor(train_data['text'].tolist())
        target_train = torch.tensor(train_data['label'].tolist())
        
        data_test = torch.tensor(test_data['text'].tolist())
        target_test = torch.tensor(test_data['label'].tolist())
        
        train_dataset = TensorDataset(data_train, target_train)
        test_dataset = TensorDataset(data_test, target_test)
        
        torch.tensor(test_data['text'].tolist())
        
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 
        return train_data_loader, test_data_loader

    def train_model(self, train_data_loader):
        self.model = Sentiment_LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, output_size=self.output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for data, labels in train_data_loader:
                data = data.to(device=self.device)
                # Ensure inputs are float and labels are long
                data = data.float()
                labels = labels.long()
        
                data = data.reshape(data.shape[0], self.seq_len, data.shape[1])
        
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss / len(train_data_loader):.4f}')
        

    def evaluate_results(self, test_data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_data_loader:
                data = data.to(device=self.device)
                
                # Ensure inputs are float and labels are long
                data = data.float()
                labels = labels.long()
        
                data = data.reshape(data.shape[0], self.seq_len, data.shape[1])
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')


if __name__=="__main__":
    lstm_obj = LSTM_Helper()
    train_data, test_data = lstm_obj.load_data()
    train_loader, test_loader = lstm_obj.data_loader(train_data, test_data)
    lstm_obj.train_model(train_loader)
    #rnn_obj.evaluate_results(train_loader)
    lstm_obj.evaluate_results(test_loader)    