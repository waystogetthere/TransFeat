import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class SANNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, dropout=0.02, num_heads=1, device="cuda"):
        super(SANNetwork, self).__init__()
        # self.fc1 = nn.Linear(input_size, input_size)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)
        self.softmax3 = nn.Softmax(dim=-1)
        self.activation = nn.SELU()
        self.num_heads = num_heads
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)
        self.multi_head = nn.ModuleList([nn.Linear(input_size, input_size) for k in range(num_heads)])


        for xx in self.multi_head:
            # nn.init.xavier_normal_(xx.weight)
            xx.weight.data.fill_(1e-3)
        # self.fc2.weight.data.fill_(0.01)
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def forward_attention(self, input_space, return_softmax=False):

        placeholder = torch.zeros(input_space.shape).to(self.device)
        for k in range(len(self.multi_head)):
            if return_softmax:
                attended_matrix = self.softmax3(self.multi_head[k](input_space))
            else:
                attended_matrix = self.softmax3(self.multi_head[k](input_space)) * input_space
                # attended_matrix = self.multi_head[k](input_space) * input_space
            placeholder = torch.add(placeholder, attended_matrix)
        placeholder /= len(self.multi_head)
        out = placeholder
        # if return_softmax:
        #    out = self.softmax(out)
        return out


    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            # print('diagonal_els', diagonal_els.detach().cpu().numpy())
            activated_diagonal = self.softmax2(diagonal_els)
            # print('activated_diagonal', activated_diagonal.detach().cpu().numpy())

            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)
        return output_mean

    def forward(self, x):

        # attend and aggregate
        out = self.forward_attention(x)
        # dense hidden (l1 in the paper)
        # out = x
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.activation(out)
        # dense hidden (l2 in the paper, output)
        # out = self.fc3(out)
        # out = self.sigmoid(out)
        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()

