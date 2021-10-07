import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FairCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, activation_fn='relu'):
        super(FairCritic, self).__init__()

        if activation_fn == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation_fn == 'leakrelu':
            act = nn.LeakyReLU(inplace=True)
        elif activation_fn == 'tanh':
            act = nn.Tanh()
        elif activation_fn == 'sigmoid':
            act = nn.Sigmoid()
        elif activation_fn == 'elu':
            act = nn.ELU(inplace=True)
        elif activation_fn == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError('Activation function: not supported yet...')

        self.model = nn.Sequential(
            *[nn.Linear(input_dim, hidden_dim), act] +
             [nn.Linear(hidden_dim, hidden_dim), act] * hidden_layers +
            [nn.Linear(hidden_dim, 1),])

    def forward(self, score, a):
        print(score.shape, a.shape)
        score_a = torch.cat((score, a), 1)
        return self.model(score_a)


class Activation_f(nn.Module):
    def __init__(self, divergence="GAN"):
        super(Activation_f, self).__init__()
        self.divergence = divergence

    def forward(self, v):
        divergence = self.divergence
        if divergence == "KLD":
            return v
        elif divergence == "RKL":
            return -torch.exp(-v)
        elif divergence == "CHI":
            return v
        elif divergence == "SQH":
            return 1-torch.exp(-v)
        elif divergence == "JSD":
            return torch.log(torch.tensor(2.))-torch.log(1.0+torch.exp(-v))
        elif divergence == "GAN":
            return -torch.log(1.0+torch.exp(-v)) # log sigmoid


class Conjugate_f(nn.Module):
    def __init__(self,divergence="GAN"):
        super(Conjugate_f,self).__init__()
        self.divergence = divergence

    def forward(self, t):
        divergence= self.divergence
        if divergence == "KLD":
            return torch.exp(t-1)
        elif divergence == "RKL":
            return -1 -torch.log(-t)
        elif divergence == "CHI":
            return 0.25*t**2+t
        elif divergence == "SQH":
            return t/(torch.tensor(1.)-t)
        elif divergence == "JSD":
            return -torch.log(2.0-torch.exp(t))
        elif divergence == "GAN":
            return  -torch.log(1.0-torch.exp(t))


class FDivLossCritic(nn.Module):
    def __init__(self,divergence="GAN"):
        super(FDivLossCritic, self).__init__()
        self.activation = Activation_f(divergence)

    def forward(self, v):
        return self.activation(v)


class FDivLossGenerator(nn.Module):
    def __init__(self,divergence="GAN"):
        super(FDivLossGenerator,self).__init__()
        self.conjugate = Conjugate_f(divergence)
        self.activation = Activation_f(divergence)

    def forward(self,v):
        return - self.conjugate(self.activation(v))


def compute_fair_loss(faircritic, fe_out, fx_out, r_sqrt_sigma, sensitive_feat, args):
    divergence = 'KLD'
    activation_f = Activation_f(divergence)
    conjugate_f = Conjugate_f(divergence)

    n_sample = args.n_train_sample if args.mode == "train" else args.n_test_sample
    n_batch = fe_out.shape[0]
    idx = np.arange(n_batch)
    np.random.shuffle(idx)
    B = r_sqrt_sigma.T.float().to(device)
    noise = torch.normal(0, 1, size=(n_sample, n_batch, args.z_dim)).to(device)
    score = torch.tensordot(noise, B, dims=1) + fx_out
    print(B.shape, noise.shape, fx_out.shape)
    
    joint = faircritic(score, sensitive_feat)
    independent = faircritic(score[:, idx, :], sensitive_feat[:, idx, :])

    loss = - activation_f(joint).mean() + conjugate_f(activation_f(independent)).mean()
    return loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    critic = FairCritic(2, 256, 1)
    critic = critic.to(device)
    divergence = 'KLD'
    activation_f = Activation_f(divergence)
    conjugate_f = Conjugate_f(divergence)

    opt = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-5)
    est_kl, real_kl = [], []
    for i in range(10000):
        a = torch.normal(1, 1, size=(256, 1)).to(device)
        b = torch.normal(1, 1, size=(256, 1)).to(device) + a * .5
        n_batch = b.shape[0]
        idx = np.arange(n_batch)
        np.random.shuffle(idx)
        joint = critic(a, b)
        independent = critic(a[idx, :], b[idx, :])
        loss = - activation_f(joint).mean() + conjugate_f(activation_f(independent)).mean()
        loss.backward()
        opt.step()

        joint_var = torch.cat((a, b), 1)
        ind_var = torch.cat((a[idx, :], b[idx, :]), 1)
        # print(joint_var.shape, ind_var.shape)
        real_kl.append(KLdivergence(joint_var.cpu(), ind_var.cpu()))
        est_kl.append(loss.item())
        if i % 500 == 0 and i:
            print(np.mean(est_kl), np.mean(real_kl))

