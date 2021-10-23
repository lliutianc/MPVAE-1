import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# class MLP(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim):
#         super(MLP, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(vocab_size, embedding_dim),
#             )
#         self.vocab_size = vocab_size
#         # out: 1 x emdedding_dim
#         self.embeddings = nn.Linear(vocab_size, embedding_dim)
#         self.linear1 = nn.Linear(embedding_dim, 128)
#         self.activation_function1 = nn.ReLU()
#
#         # out: 1 x vocab_size
#         self.linear2 = nn.Linear(128, vocab_size)
#         self.activation_function2 = nn.LogSoftmax(dim=-1)
#
#     def forward(self, inputs):
#         # bsz = inputs.shape[0]
#         # assert inputs.shape[1] == self.vocab_size, 'invalid inpu shape does not matcht'
#         # idx = torch.stack([torch.arange(self.vocab_size) for _ in range(bsz)])
#         # inputs = idx[inputs == 1]
#
#         embeds = self.embeddings(inputs)
#         out = self.linear1(embeds)
#         out = self.activation_function1(out)
#         out = self.linear2(out)
#         out = self.activation_function2(out)
#         return out
#
#     def get_emdedding(self, label):
#         word = torch.tensor([label])
#         return self.embeddings(word).view(1, -1)


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=None):
        super(CBOW, self).__init__()

        # out: 1 x emdedding_dim
        hidden_dim = hidden_dim or int((vocab_size + embedding_dim) / 2)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.sum(-2)
        if len(embeds.shape) == 1:
            embeds = embeds.view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)

        return out

    def get_embedding(self, inputs):
        return self.embeddings(inputs).sum(-2)


class CBOWData(Dataset):
    def __init__(self, labels, device='cpu'):
        super(CBOWData, self).__init__()

        idx = np.arange(labels.shape[1])
        self.context = []
        self.target = []

        for label in labels:
            act_label = idx[label == 1]
            for i in range(len(act_label)):
                target = act_label[[i]]
                context = np.concatenate(
                    [act_label[:max(0, i-1)], act_label[min(i+1, len(act_label)):]])
                self.target.append(torch.from_numpy(target))
                self.context.append(torch.from_numpy(context))

        self.device = device

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return self.context[item].to(self.device), self.target[item].to(self.device)



