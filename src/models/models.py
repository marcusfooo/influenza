
from torch import nn, zeros
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
  def __init__(self, input_dim, hidden_size, attention_size, num_of_layers, output_dim):
    super(AttentionModel, self).__init__()

    self.hidden_size = hidden_size
    self.num_of_layers = num_of_layers
    
    self.layer1 = nn.LSTM(input_dim, hidden_size, num_of_layers)
    self.att_w1 = nn.Linear(hidden_size, attention_size)
    self.att_w2 = nn.Linear(attention_size, attention_size)
    self.att_v = nn.Linear(attention_size, 1)
    self.fc_layer = nn.Linear(hidden_size, output_dim)

  def forward(self, input_seq, hidden_state):
    out, h, c = self.layer1(input_seq, hidden_state)
    x = self.attention(out, h)


  def attention(self, features, h):
    score = F.tanh(self.att_w1(features) + self.att_w2(h))
    attention_weights = nn.functional.softmax(self.att_v(score), dim=1)
    context_vec = attention_weights * features
    #context_vec = context_vec.sum()
    #context_vec = tf.reduce_sum(context_vector, axis=1)

    return context_vec, attention_weights