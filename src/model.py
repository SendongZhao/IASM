import torch

import math

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias = True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



class MultiLayer(nn.Module):
    def __init__(self, n_in_feature, n_hidden, n_out_feature, dropout):
        super(MultiLayer, self).__init__()

        self.gc1 = GraphConvolution(n_in_feature, n_out_feature)
        #self.gc2 = GraphConvolution(n_hidden, n_out_feature)

        self.dropout = dropout

    def forward(self, input, adj, adj_trans):

        output = F.relu(self.gc1(input, adj))

        #output = F.dropout(output, self.dropout, training=self.training)

        #output = F.relu(self.gc2(output, adj_trans))

        #output = F.dropout(output, self.dropout, training=self.training)

        return output


class Scoring(nn.Module):

    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(Scoring, self).__init__()

        #print ("Initialize Scoring ...")

        self.gcn_query = MultiLayer(in_feature, hidden, out_feature, dropout)
        self.gcn_document = MultiLayer(in_feature, hidden, out_feature, dropout)

    def distance(self, A, B):
        """
        you can use different distance measures
        :param A:
        :param B:
        :return: distance
        """
        A = A/torch.norm(A)
        B = B/torch.norm(B)
        
        return torch.dist(A, B)


    def forward(self, query, document, adj, alpha, beta):

        adj_trans = torch.transpose(adj, 1, 0)

        #print ("adj_trans, query")
        #print (adj_trans.size())
        #print (query.size())
        convo_query = self.gcn_query(query, adj_trans, adj)
        #print ("Got new vector for query after convolution ...")
        #print (convo_query.size())
        #print ("Convoluted query")
        #print (convo_query)

        convo_document = self.gcn_document(document, adj, adj_trans)
        #print ("Got new vector for document after convolution ...")
        #print (convo_document.size())
        #print ("Convoluted document")
        #print (convo_document)

        dist_query = self.distance(query, convo_document)
        #print ("dist_query: ")
        #print (dist_query)

        dist_document = self.distance(document, convo_query)
        #print ("dist_document:")
        #print (dist_document)

        dist = alpha*dist_query + beta*dist_document

        return dist


