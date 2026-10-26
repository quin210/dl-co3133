from torch import nn

class LSTMClassifier(nn.Module):
    # image rows form a sequence from top to bottom
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        rows = images.squeeze(1)
        _, (hidden, _) = self.lstm(rows)
        return self.classifier(hidden[-1])

class LSTMColumns(nn.Module):
    # image columns form a sequence from left to right
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        columns = images.squeeze(1).transpose(1, 2)
        _, (hidden, _) = self.lstm(columns)
        return self.classifier(hidden[-1])
