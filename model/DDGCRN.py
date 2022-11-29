import torch
import torch.nn as nn
from model.DDGCRNCell import DDGCRNCell
class DGCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(DDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.DGCRM_cells.append(DDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :], node_embeddings[1]])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class DDGCRN(nn.Module):
    def __init__(self, args):
        super(DDGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))

        self.encoder1 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        self.encoder2 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        #predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
    def forward(self, source, i=2):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # init_state = self.encoder.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        # output = self.dropout1(output[:, -1:, :, :])
        #
        # #CNN based predictor
        # output = self.end_conv((output))                         #B, T*C, N, 1
        #
        # return output
        node_embedding1 = self.node_embeddings1
        if self.use_D:
            t_i_d_data   = source[..., 1]

            # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data   = source[..., 2]
            # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        # time_embeddings = T_i_D_emb
        # time_embeddings = D_i_W_emb

        node_embeddings=[node_embedding1,self.node_embeddings1]

        source = source[..., 0].unsqueeze(-1)

        if i == 1:
            init_state1 = self.encoder1.init_hidden(source.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output, _ = self.encoder1(source, init_state1, node_embeddings)  # B, T, N, hidden
            # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
            output = self.dropout1(output[:, -1:, :, :])

            # CNN based predictor
            output1 = self.end_conv1(output)  # B, T*C, N, 1

            return output1

        else:
            init_state1 = self.encoder1.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
            output, _ = self.encoder1(source, init_state1, node_embeddings)      #B, T, N, hidden
            # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
            output = self.dropout1(output[:, -1:, :, :])

            #CNN based predictor
            output1 = self.end_conv1(output)                         #B, T*C, N, 1

            source1 = self.end_conv2(output)

            source2 = source -source1

            init_state2 = self.encoder2.init_hidden(source2.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
            output2, _ = self.encoder2(source2, init_state2, node_embeddings)      #B, T, N, hidden
            # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
            output2 = self.dropout2(output2[:, -1:, :, :])

            # source2 = self.end_conv4(output2)

            output2 = self.end_conv3(output2)

            return output1 + output2



        # # source3 = source - source1 - source2
        # # init_state3 = self.encoder3.init_hidden(source3.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
        # # output3, _ = self.encoder3(source3, init_state3, node_embeddings)      #B, T, N, hidden
        # # # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
        # # output3 = self.dropout3(output3[:, -1:, :, :])
        # #
        # # output3 = self.end_conv5(output3)
        #
        # # return output1+output2+output3

# class AGCRN(nn.Module):
#     def __init__(self, args):
#         super(AGCRN, self).__init__()
#         self.num_node = args.num_nodes
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.rnn_units
#         self.output_dim = args.output_dim
#         self.horizon = args.horizon
#         self.num_layers = args.num_layers
#         self.dropout1 = nn.Dropout(p=0.1)
#         self.default_graph = args.default_graph
#         self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
#         # self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#         #                         args.embed_dim, args.num_layers)
#
#         self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                  args.embed_dim, args.num_layers)
#         # predictor
#         self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#
#         # self.end_conv5 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#
#     def forward(self, source, targets, teacher_forcing_ratio=0.5):
#         # source: B, T_1, N, D
#         # target: B, T_2, N, D
#         #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
#
#         init_state = self.encoder.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
#         output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
#         # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
#         output = self.dropout1(output[:, -1:, :, :])
#
#         #CNN based predictor
#         output = self.end_conv((output))                         #B, T*C, N, 1
#
#
#         return output


# class AGCRN(nn.Module):
#     def __init__(self, args):
#         super(AGCRN, self).__init__()
#         self.num_node = args.num_nodes
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.rnn_units
#         self.output_dim = args.output_dim
#         self.horizon = args.horizon
#         self.num_layers = args.num_layers
#         self.dropout1 = nn.Dropout(p=0.1)
#         self.dropout2 = nn.Dropout(p=0.1)
#         self.dropout3 = nn.Dropout(p=0.1)
#         self.default_graph = args.default_graph
#         self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
#
#         self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                 args.embed_dim, args.num_layers)
#         self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                 args.embed_dim, args.num_layers)
#         self.encoder3 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                 args.embed_dim, args.num_layers)
#         #predictor
#         self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#
#         self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#         self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#         self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#         self.end_conv4 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#         self.end_conv5 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#     def forward(self, source, targets, teacher_forcing_ratio=0.5):
#         #source: B, T_1, N, D
#         #target: B, T_2, N, D
#         # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
#
#         # init_state = self.encoder.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
#         # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
#         # # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
#         # output = self.dropout1(output[:, -1:, :, :])
#         #
#         # #CNN based predictor
#         # output = self.end_conv((output))                         #B, T*C, N, 1
#         #
#         #
#         # return output
#
#
#
#         init_state = self.encoder1.init_hidden(source.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
#         output, _ = self.encoder1(source, init_state, self.node_embeddings)      #B, T, N, hidden
#         # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
#         output = self.dropout1(output[:, -1:, :, :])
#
#         #CNN based predictor
#         source1 = self.end_conv1(output)
#         output = self.end_conv2(output)                         #B, T*C, N, 1
#
#
#
#         source1 = source -source1
#         init_state1 = self.encoder2.init_hidden(source1.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
#         output1, _ = self.encoder2(source1, init_state1, self.node_embeddings)      #B, T, N, hidden
#         # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
#         output1 = self.dropout2(output1[:, -1:, :, :])
#
#         source2 = self.end_conv3(output1)
#         output1 = self.end_conv4(output1)
#
#         source2 = source1 -source2
#         init_state2 = self.encoder3.init_hidden(source2.shape[0])   #[2,64,307,64] 前面是2是因为有两层GRU
#         output2, _ = self.encoder3(source2, init_state2, self.node_embeddings)      #B, T, N, hidden
#         # output2 = output2[:, -1:, :, :]                                   #B, 1, N, hidden
#         output2 = self.dropout3(output2[:, -1:, :, :])
#
#         output2 = self.end_conv5(output2)
#
#
#         return output + output1 + output2