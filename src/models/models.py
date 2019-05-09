
from torch import nn, zeros, bmm, squeeze, unsqueeze, tanh, cat
import torch.nn.functional as F

class GruModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, hidden_size, output_dim):
    super(GruModel, self).__init__()
  
    self.hidden_size = hidden_size
    self.output_dim = output_dim
    
    self.layer1 = nn.GRU(input_dim, hidden_size)
    self.layer2 = nn.Linear(hidden_size, output_dim)

  def forward(self, input_seq, hidden_state):
    out, _ = self.layer1(input_seq, hidden_state)
    score_seq = self.layer2(out[-1,:,:])

    return score_seq
  
  def init_hidden(self, batch_size):
    return zeros(1, batch_size, self.hidden_size)


class LstmModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, hidden_size, output_dim):
    super(LstmModel, self).__init__()
  
    self.hidden_size = hidden_size
    self.output_dim = output_dim
    
    self.layer1 = nn.LSTM(input_dim, hidden_size)
    self.layer2 = nn.Linear(hidden_size, output_dim)

  def forward(self, input_seq, hidden_state):
    out, _ = self.layer1(input_seq, hidden_state)
    score_seq = self.layer2(out[-1,:,:])

    return score_seq
  
  def init_hidden(self, batch_size):
    h_init = zeros(1, batch_size, self.hidden_size)
    c_init = zeros(1, batch_size, self.hidden_size)
    
    return (h_init, c_init)


class AttentionModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, seq_length, hidden_size, output_dim):
    super(AttentionModel, self).__init__()

    self.hidden_size = hidden_size
    self.seq_length = seq_length
    self.output_dim = output_dim
    
    self.encoder = nn.LSTM(input_dim, hidden_size)
    self.attn = nn.Linear(hidden_size, seq_length)
    self.dropout = nn.Dropout()
    self.out = nn.Linear(hidden_size, output_dim)


  def forward(self, input_seq, hidden_state):
    encoder_outputs, (h, _) = self.encoder(input_seq, hidden_state)
    attn_applied, attn_weights = self.attention(encoder_outputs, h)
    attn_applied = self.dropout(attn_applied)
    score_seq = self.out(attn_applied.reshape(-1, self.hidden_size))

    return score_seq

  def attention(self, encoder_outputs, hidden):
    attn_weights = F.softmax(squeeze(self.attn(hidden)), dim=1)
    attn_weights = unsqueeze(attn_weights, 1)
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    attn_applied = bmm(attn_weights, encoder_outputs)

    return attn_applied, attn_weights

  def init_hidden(self, batch_size):
    h_init = zeros(1, batch_size, self.hidden_size)
    c_init = zeros(1, batch_size, self.hidden_size)
    
    return (h_init, c_init)


class DaRnnModel(nn.Module):
  """TODO: DOCSTRING"""
  def __init__(self, input_dim, seq_length, hidden_size, output_dim):
    super(DaRnnModel, self).__init__()

    self.n = input_dim
    self.m = hidden_size
    self.T = seq_length
    self.output_dim = output_dim
    
    self.encoder = nn.LSTM(self.n, self.m)

    self.We = nn.Linear(2 * self.m, self.T)
    self.Ue = nn.Linear(self.T, self.T)
    self.ve = nn.Linear(self.T, 1)

    self.Ud = nn.Linear(self.m, self.m)
    self.vd = nn.Linear(self.m, 1)
    self.out = nn.Linear(self.m, output_dim)

  def forward(self, x, hidden_state):
    h_seq = []
    for t in range(self.T):
      x_tilde, _ = self.input_attention(x, hidden_state, t)
      ht, hidden_state = self.encoder(x_tilde, hidden_state)
      h_seq.append(ht)

    h = cat(h_seq, dim=0)
    attn_applied, attn_weights = self.temporal_attention(h)
    score_seq = self.out(attn_applied)

    return score_seq

  def input_attention(self, x, hidden_state, t):
    x = x.permute(1, 2, 0)
    h, c = hidden_state
    h = h.permute(1, 0, 2)
    c = c.permute(1, 0, 2)
    hc = cat([h, c], dim=2)

    e = self.ve(tanh(self.We(hc) + self.Ue(x)))
    e = squeeze(e)
    alpha = F.softmax(e, dim=1)
    xt = x[:, :, t]

    x_tilde = alpha * xt
    x_tilde = unsqueeze(x_tilde, 0)

    return x_tilde, alpha

  def temporal_attention(self, h):
    h = h.permute(1, 0, 2)
    l = self.vd(tanh((self.Ud(h))))
    l = squeeze(l)
    beta = F.softmax(l, dim=1)
    beta = unsqueeze(beta, 1)
    c = bmm(beta, h)
    c = squeeze(c)

    return c, beta

  def init_hidden(self, batch_size):
    h_init = zeros(1, batch_size, self.m)
    c_init = zeros(1, batch_size, self.m)
    
    return (h_init, c_init)

