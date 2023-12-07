# -*- coding: utf-8 -*-
"""MidiGenTransformer.ipynb

Original file is located at
    https://colab.research.google.com/drive/1WF7f_kffaQY0rHKMi8UXtDexX4twqnGj

##0. Installation
"""

!pip install d2l

import torch
from d2l import torch as d2l

!pip install pretty_midi # to process midi-file
!pip install torch torchvision

import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import Transformer

from google.colab import drive
import os

drive.mount('/content/drive') # your drive mount

my_drive_path = '/content/drive/MyDrive/been_colab/MIDI-Transformer/data/umm' # your path to midi-files
contents = os.listdir(my_drive_path)
print(contents)

"""##1. Loading and preprocessing midi-files

###1.1. Loading midi file with pretty_midi Library
"""

def load_midi_files(midi_folder):
    midi_data = []
    for file in os.listdir(midi_folder):
        if file.endswith(".mid") or file.endswith(".midi"):
            file_path = os.path.join(midi_folder, file)
            try:
                midi = pretty_midi.PrettyMIDI(file_path)
                midi_data.append(midi)
            except IOError:
                print(f"Cannot load {file}")
            except ValueError as e:
                # skip the file if an error occurs
                print(f"Skipping corrupted MIDI file: {file} - {e}")
    return midi_data

midi_files = load_midi_files(my_drive_path) # midi_folder = my_drive_path

midi_files[0] # to check if loading succeeded

"""###1.2. EDA (to identify what features in a midi file)"""

for i, midi in enumerate(midi_files):
    print(f"MIDI file {i}:")
    print(f"  Number of instruments: {len(midi.instruments)}")
    print(f"  Length in seconds: {midi.get_end_time()}")
    print(f"  Number of notes: {len(midi.instruments[0].notes)}")
    print(f"  First note: {midi.instruments[0].notes[0]}")
    print(f"  Last note: {midi.instruments[0].notes[-1]}")

"""###1.3. Preprocessing"""

def tokenize_notes(midi_file): # Tokenize "the first instrument line" in all MIDI files "by notes"
    tokens = []
    for note in midi_file.instruments[0].notes:
        token = []
        token.append(note.start)
        token.append(note.end)
        token.append(note.pitch)
        token.append(note.velocity)
        token_tensor = torch.tensor(token)
        tokens.append(token_tensor) # Append the information(start time, end time, pitch, velocity) of each note as a tensor.
    return tokens

all_midi_tokens = []

for midi_file in midi_files:
  one_midi_tokens = tokenize_notes(midi_file)
  all_midi_tokens.append(one_midi_tokens)

print("# of midi dataset: ", len(all_midi_tokens))
for i in range(len(all_midi_tokens)):
  print("# of tokens of midi: ", len(all_midi_tokens[i])) # to check how many notes(of the first instrument) in a midi file

"""##2. Model (w/ Attention and Transformer!)

###2.1. Train / Validation / Test data split
"""

class DataSplitter:
    def __init__(self, tensors, train_percent=0.7, val_percent=0.2, test_percent=0.1):
        self.tensors = tensors
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent

    def split(self):
        total_size = len(self.tensors)
        train_size = int(total_size * self.train_percent)
        val_size = int(total_size * self.val_percent)

        train_data = self.tensors[:train_size]
        val_data = self.tensors[train_size:train_size + val_size]
        test_data = self.tensors[train_size + val_size:]

        return train_data, val_data, test_data

splitter = DataSplitter(all_midi_tokens)
train_data, val_data, test_data = splitter.split()

print("# of train data point: ", len(train_data))
print("# of validation data point: ", len(val_data))
print("# of test data point: ", len(test_data))

"""###2.2. Dataset Class"""

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, sequence_length=30):
        """
        data: a list of PyTorch tensors comprising the dataset
        sequence_length: length of input sequence and target sequence (should be same! .. or not)
        """
        self.data = data
        self.sequence_length = sequence_length # default : 30

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]  # whole sequence (before split into input sequence and output sequence)

        # identify length of input sequence and output sequence
        input_length = min(len(sequence), self.sequence_length)
        target_length = min(len(sequence) - input_length, self.sequence_length)

        # split a whole sequence into input sequence and output sequence
        input_sequence = sequence[:input_length]
        target_sequence = sequence[input_length:input_length + target_length]

        # "zero(torch.tensor([0, 0, 0, 0]) padding" if needed
        if input_length < self.sequence_length:
            while len(input_sequence) < self.sequence_length :
                input_sequence.append(torch.tensor([0, 0, 0, 0])) # padding to make the number of tokens be 30
        if target_length < self.sequence_length:
            while len(target_sequence) < self.sequence_length:
                target_sequence.append(torch.tensor([0, 0, 0, 0])) # padding to make the number of tokens be 30

        return torch.stack(input_sequence).to(torch.float32), torch.stack(target_sequence).to(torch.float32) # torch.float64 -> torch.float32 to use at Transformer model

# create a CustomDataset instance for Train, Validation, and Test datasets
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)
test_dataset = CustomDataset(test_data)

# generate batches for training and evaluation using PyTorch DataLoader
from torch.utils.data import DataLoader

# batch_size is a hyper-parameter! can be changed
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

"""###2.3. Model Class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(4, d_model)  # transform four features into d_model dimensions
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(d_model, 4)  # restore the output to its original four features

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# generate model instance
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_len = 30
model = TransformerModel(d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_len)

# definition of loss function and optimizer
criterion = torch.nn.MSELoss() # hyper..
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # hyper..

"""###2.4. Training and Evaluation"""

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for input_sequence, target_sequence in train_loader:
        input_sequence = input_sequence
        target_sequence = target_sequence

        optimizer.zero_grad()
        output = model(input_sequence)
        loss = criterion(output, target_sequence)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch}/{num_epochs} - Training Loss: {avg_train_loss}')

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_sequence, target_sequence in val_loader:
            output = model(input_sequence)
            loss = criterion(output, target_sequence)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Epoch {epoch}/{num_epochs} - Validation Loss: {avg_val_loss}')
