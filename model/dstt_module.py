# Decoupled spatial-temporal transformer for video inpainting (arXiv 2021)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class TransformerBlock(nn.Module):
#     """
#     Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
#     """
#
#     def __init__(self, tokensize, hidden=128, num_head=4, mode='s', dropout=0.1):
#         super().__init__()
#         self.attention = MultiHeadedAttention(tokensize, d_model=hidden, head=num_head, mode=mode, p=dropout)
#         self.ffn = FeedForward(hidden, p=dropout)
#         self.norm1 = nn.LayerNorm(hidden)
#         self.norm2 = nn.LayerNorm(hidden)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, input):
#         x, t = input['x'], input['t']
#         x = self.norm1(x)
#         x = x + self.dropout(self.attention(x, t))
#         y = self.norm2(x)
#         x = x + self.ffn(y)
#         return {'x': x, 't': t}


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, token_size, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = token_size

    def forward(self, x, t):
        bt, n, c = x.size()
        b = bt // t
        c_h = c // self.head
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        if self.mode == 's':
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)
            att, _ = self.attention(query, key, value)
            att = att.permute(0, 1, 3, 2, 4).view(bt, n, c)
        elif self.mode == 't':
            key = key.view(b, t, 2, self.h // 2, 2, self.w // 2, self.head, c_h)
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            query = query.view(b, t, 2, self.h // 2, 2,
                               self.w // 2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            value = value.view(b, t, 2, self.h // 2, 2,
                               self.w // 2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).view(
                b, 4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.view(b, 2, 2, self.head, t, self.h // 2, self.w // 2, c_h)
            att = att.permute(0, 4, 1, 5, 2, 6, 3,
                              7).view(bt, n, c)
        output = self.output_linear(att)
        return output


def main():
    attention_block_s = MultiHeadedAttention(
        token_size=[4, 8], d_model=64, head=4, mode='s')
    attention_block_t = MultiHeadedAttention(
        token_size=[4, 8], d_model=64, head=4, mode='t')

    inputs = torch.rand([8, 32, 64])
    print(inputs.size())
    # inputs = torch.rand(1, 3, 256, 256)

    output = attention_block_s(inputs, 2)
    output = attention_block_t(output, 2)
    print(inputs.size(), output.size())


if __name__ == '__main__':
    main()
