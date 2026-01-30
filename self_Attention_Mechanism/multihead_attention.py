class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)
    
    def forward(self, query, key, value):
        return self.attention(query, key, value)
    

    