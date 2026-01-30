class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=8),
            num_layers=num_layers
        )
    
    def forward(self, x):
        x = self.embedding(x)
        return self.transformer(x, x)