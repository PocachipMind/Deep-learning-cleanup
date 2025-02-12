import torch


class Rnn(torch.nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, n_class
    ) -> None:
        super(Rnn, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = self.embedding(x)
        x, h_N = self.rnn(x)
        x = torch.sum(h_N, axis=0)
        x = self.fc(x)
        return x


class Gru(torch.nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, n_class
    ) -> None:
        super(Gru, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.gru = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        x = self.embedding(x)
        x, h_N = self.gru(x)
        x = torch.sum(h_N, axis=0)
        x = self.fc(x)
        return x


class Lstm(torch.nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, n_class
    ) -> None:
        super(Lstm, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, n_class)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, (h_N, c_N) = self.lstm(x)
        x = torch.sum(h_N, axis=0)
        x = self.fc(x)
        return x
