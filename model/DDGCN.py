'''
这个例子对应最后的x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
简单来说就是多项式每项与权重相乘后相加
x=np.random.randint(0, high=100, size=[4,6,2,8])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[6,2,8,16])
#y=torch.randn(6,2,8,16)
y=torch.from_numpy(y)
out=torch.einsum('bnki,nkio->bno', x, y)
out1=torch.einsum('bni,nio->bno', x[:,:,0,:],y[:,0,:,:])+torch.einsum('bni,nio->bno', x[:,:,1,:],y[:,1,:,:])
print(np.allclose(out.numpy(), out1.numpy()))
'''

'''
就是d这个维度的元素相乘相加
weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)

x=np.random.randint(0, high=100, size=[4,5])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[5,2,6,8])
#y=torch.randn(6,2,8,16)
y=torch.from_numpy(y)
weights1 = torch.einsum('nd,dkio->nkio', x, y)[:,:,1,1]
weights2 = torch.einsum('nd,dk->nk', x, y[:,:,1,1])
print(np.allclose(weights1.numpy(), weights2.numpy()))
'''

'''
x=np.random.randint(0, high=100, size=[2,5,5])
x=torch.from_numpy(x)
y=np.random.randint(0, high=100, size=[6,5,8])
y=torch.from_numpy(y)
x_g = torch.einsum("knm,bmc->bknc", x, y)
x1= torch.einsum("nm,bmc->bnc", x[0], y)+torch.einsum("nm,bmc->bnc", x[1], y)
x2=x_g[:,0,:,:]+x_g[:,1,:,:]
print(np.allclose(x1.numpy(), x2.numpy()))
'''

'''
def get_laplacian(graph, I, normalize=True):
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    if normalize:
        D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = I - torch.matmul(torch.matmul(D, graph), D)
    else:
        D = torch.diag(torch.sum(graph, dim=-1))
        L = D - graph
    return L

supports1 = torch.eye(10).cuda()

D1 = torch.ones([10,1])
D2 = torch.ones([1,10])
x = F.relu(torch.randn([10,10,10]).cuda())
x1 = torch.sum(x, dim=-1) ** (-1 / 2)
x2 = torch.diag_embed(x1)

y = get_laplacian(x, supports1)

#D = torch.einsum('nm,mc->nc', D1, D2)
print(y)

'''




import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class DGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential( #疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             #('sigmoid1', nn.ReLU()),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings[0].shape[1]
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)
        filter = self.fc(x)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  #[B,N,dim_in]
        supports2 = DGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)

        #supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)


        #default cheb_k = 3
        # for k in range(2, self.cheb_k):
        #     support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        #supports3 = torch.matmul(2 * supports2, supports2) - supports1
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        #x_g3 = torch.einsum("bnm,bmc->bnc", supports3, x)
        x_g = torch.stack([x_g1,x_g2],dim=1)

        # supports = torch.stack(support_set, dim=0)   #[2,nodes,nodes]  也就是这里把单位矩阵和自适应矩阵拼在一起了
        # x_g = torch.einsum("knm,bmc->bknc", supports, x)

        # weights = torch.einsum('bnd,dkio->bnkio', nodevec, self.weights_pool)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)    #[B,N,embed_dim]*[embed_dim,chen_k,dim_in,dim_out] =[B,N,cheb_k,dim_in,dim_out]
                                                                                  #[N, cheb_k, dim_in, dim_out]=[nodes,cheb_k,hidden_size,output_dim]
        bias = torch.matmul(node_embeddings[1], self.bias_pool) #N, dim_out                 #[che_k,nodes,nodes]* [batch,nodes,dim_in]=[B, cheb_k, N, dim_in]

        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_gconv = torch.einsum('bnki,bnkio->bno', x_g, weights) + bias  #b, N, dim_out
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  #b, N, dim_out
        # x_gconv = torch.einsum('bnki,kio->bno', x_g, self.weights) + self.bias    #[B,N,cheb_k,dim_in] *[N,cheb_k,dim_in,dim_out] =[B,N,dim_out]

        return x_gconv

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            #L = I - torch.matmul(torch.matmul(D, graph), D)
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L
