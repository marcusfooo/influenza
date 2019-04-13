
from torch import nn, zeros

class LSTMModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, hidden_size, num_of_layers, output_dim):
    super(LSTMModel, self).__init__()
  
    self.hidden_size = hidden_size
    self.num_of_layers = num_of_layers
    
    self.layer1 = nn.LSTM(input_dim, hidden_size, num_of_layers)
    self.layer2 = nn.Linear(hidden_size, output_dim)


  def forward(self, trigram_seq, hidden_state):
    LSTM_output, _ = self.layer1(trigram_seq, hidden_state)
    score_seq = self.layer2(LSTM_output[-1,:,:])

    return score_seq
  
  
  def init_hidden(self, batch_size):
    h_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    c_init = zeros(self.num_of_layers, batch_size, self.hidden_size)
    
    return (h_init, c_init)