
from torch import nn, zeros, bmm
import torch.nn.functional as F

class LSTMModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, hidden_size, num_of_layers, output_dim):
    super(LSTMModel, self).__init__()
  
    self.hidden_size = hidden_size
    self.num_of_layers = num_of_layers
    
    self.layer1 = nn.LSTM(input_dim, hidden_size, num_of_layers)
    self.layer2 = nn.Linear(hidden_size, output_dim)


  def forward(self, input_seq, hidden_state):
    out, _ = self.layer1(input_seq, hidden_state)
    score_seq = self.layer2(out[-1,:,:])

    return score_seq
  
  
  def init_hidden(self, batch_size):
    h_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    c_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    
    return (h_init, c_init)


class AttentionModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, seq_length, hidden_size, num_of_layers, output_dim):
    super(AttentionModel, self).__init__()

    self.hidden_size = hidden_size
    self.seq_length = seq_length
    self.num_of_layers = num_of_layers
    
    self.encoder = nn.LSTM(input_dim, hidden_size, num_of_layers)
    self.attn = nn.Linear(hidden_size, seq_length)
    self.out = nn.Linear(hidden_size, output_dim)


  def forward(self, input_seq, hidden_state):
    encoder_outputs, (h, _) = self.encoder(input_seq, hidden_state)
    attn_applied, attn_weights = self.attention(encoder_outputs, h)
    score_seq = self.out(attn_applied.view(-1, self.hidden_size))

    return score_seq


  def attention(self, encoder_outputs, hidden):
    attn_weights = F.softmax(self.attn(hidden[-1]), dim=1)
    attn_weights = attn_weights.view(-1, 1, self.seq_length)
    encoder_outputs = encoder_outputs.view(-1, self.seq_length, self.hidden_size)
    attn_applied = bmm(attn_weights, encoder_outputs)

    return attn_applied, attn_weights


  def init_hidden(self, batch_size):
    h_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    c_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    
    return (h_init, c_init)