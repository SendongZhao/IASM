import torch
import torch.nn as nn
from torch.autograd import Variable


class marginLoss(nn.Module):
    def __init__(self):
        super(marginLoss, self).__init__()

    def forward(self, true, fake, margin):
        zero_tensor = torch.FloatTensor(true.size()).cuda()
        zero_tensor.zero_()
        zero_tensor = Variable(zero_tensor)
        return torch.sum(torch.max(true - fake + margin, zero_tensor))