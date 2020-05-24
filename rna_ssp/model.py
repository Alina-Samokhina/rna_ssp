import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, emb_dim, vocab_size, output_dim, dropout = 0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(vocab_size, emb_dim) 
        
        self.conv1 = nn.Conv1d(
            in_channels=self.emb_dim, 
            out_channels=output_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )


        self.conv2 = nn.Conv1d(
            in_channels=self.emb_dim*2, 
            out_channels=self.emb_dim, 
            kernel_size=7, 
            stride=1, 
            padding=3
        )
        
        self.conv3 = nn.Conv1d(
            in_channels=emb_dim, 
            out_channels=self.emb_dim*2, 
            kernel_size=11, 
            stride=1, 
            padding=5
        )
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq):
        x = self.emb(seq)
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1(x)  
        out = x.permute(0, 2, 1)
        return out