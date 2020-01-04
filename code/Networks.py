import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, device, input_size, hidden_size, diagnoses_count, procedures_count, n_layers=1,
                 embedding_dropout_rate=0, gru_dropout_rate=0, embedding_diagnoses_np=None,
                 embedding_procedures_np=None, bidirectional=False):
        super(Encoder, self).__init__()

    def forward(self, input_medication, input_diagnoses, input_procedures, hidden_diagnoses=None,
                hidden_procedures=None):
        pass

    def initialize_weights(self):
        init_range = 0.1
        self.embedding_diagnoses.weight.data.uniform_(-init_range, init_range)
        self.embedding_procedures.weight.data.uniform_(-init_range, init_range)


class EncoderRNNQuery(Encoder):
    def __init__(self, device, input_size, hidden_size, diagnoses_count, procedures_count, n_layers=1,
                 embedding_dropout_rate=0, gru_dropout_rate=0, embedding_diagnoses_np=None,
                 embedding_procedures_np=None, bidirectional=False):
        super(EncoderRNNQuery, self).__init__(device, input_size, hidden_size, diagnoses_count, procedures_count,
                                              n_layers, embedding_dropout_rate, gru_dropout_rate,
                                              embedding_diagnoses_np, embedding_procedures_np, bidirectional)

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_diagnoses = nn.Embedding(diagnoses_count, input_size)
        self.embedding_procedures = nn.Embedding(procedures_count, input_size)
        self.n_layers = n_layers
        self.embedding_dropout_rate = embedding_dropout_rate
        self.gru_dropout_rate = gru_dropout_rate
        self.bidirectional = bidirectional
        self.gru_diagnoses = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                    dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                    bidirectional=self.bidirectional)
        self.gru_procedures = nn.GRU(self.input_size, self.hidden_size, self.n_layers,
                                     dropout=(0 if self.n_layers == 1 else self.gru_dropout_rate),
                                     bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.embedding_dropout_rate)
        if bidirectional:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * 2 * hidden_size, 2 * hidden_size), nn.ReLU(),
                                                  nn.Linear(2 * hidden_size, hidden_size))
        else:
            self.linear_embedding = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden_size, hidden_size))
        self.gru_representation = nn.GRU(hidden_size, hidden_size)
        self.initialize_weights()

        if embedding_diagnoses_np is not None:  # use pretrained embedding vectors to initialize the embeddings
            print('use pretrained embedding vectors to initialize diagnoses embeddings')
            self.embedding_diagnoses.weight.data.copy_(torch.from_numpy(embedding_diagnoses_np))
        if embedding_procedures_np is not None:
            print('use pretrained embedding vectors to initialize procedures embeddings')
            self.embedding_procedures.weight.data.copy_(torch.from_numpy(embedding_procedures_np))

    def forward(self, input_medication, input_diagnoses, input_procedures, hidden_diagnoses=None,
                hidden_procedures=None):

        seq_diagnoses = []
        seq_procedures = []
        for admission in input_diagnoses:
            data = self.dropout(self.embedding_diagnoses(torch.LongTensor(admission).to(self.device))).mean(
                dim=0, keepdim=True)
            seq_diagnoses.append(data)
        for admission in input_procedures:
            data = self.dropout(self.embedding_diagnoses(torch.LongTensor(admission).to(self.device))).mean(
                dim=0, keepdim=True)
            seq_procedures.append(data)

        seq_diagnoses = torch.cat(seq_diagnoses).unsqueeze(dim=1)  # dim=(#admission,1,input_size)
        seq_procedures = torch.cat(seq_procedures).unsqueeze(dim=1)  # dim=(#admission,1,input_size)

        # output dim=(#admission,1,num_direction*hidden_size)
        # hidden dim=(num_layers*num_directions,1,hidden_size)
        output_diagnoses, hidden_diagnoses = self.gru_diagnoses(seq_diagnoses)
        output_procedures, hidden_procedures = self.gru_procedures(seq_procedures)
        patient_representations = torch.cat([output_diagnoses, output_procedures],
                                            dim=-1)  # dim=(#admission,1,2*hidden_size*num_direction)
        linear_representation = self.linear_embedding(patient_representations)  # dim=(#admission,1, hidden_size)

        queries, _ = self.gru_representation(linear_representation)  # query dim=(#admission,1,hidden_size)
        queries = queries.squeeze(dim=1)  # query dim=(#admission,hidden_size)
        query = queries[-1:]  # representation of the last admission, dim=(1, hidden_size)

        if len(input_diagnoses) > 1:
            memory_keys = queries[:-1]  # dim=(#admission-1,hidden_size)
            memory_values = input_medication[:-1]  # a list of list, medications except for the last admission
        else:
            memory_keys = None
            memory_values = None

        return query, memory_keys, memory_values


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method, choose from dot, general, and concat.")

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    # score=query.T()*keys
    def dot_score(self, query, keys):
        return torch.sum(query * keys, -1).unsqueeze(0)  # dim=(1,keys.dim(0))

    # score=query.T()*W*keys, W is a matrix
    def general_score(self, query, keys):
        energy = self.attn(keys)
        return torch.sum(query * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0))

    # score=v.T()*tanh(W*[query;keys])
    def concat_score(self, query, keys):
        energy = self.attn(torch.cat((query.expand(keys.size(0), -1), keys), -1)).tanh()
        return torch.sum(self.v * energy, -1).unsqueeze(0)  # dim=(1, keys.dim(0)

    def initialize_weights(self, init_range):
        if self.method == 'concat':
            self.v.data.uniform_(-init_range, init_range)

    def forward(self, query, keys):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(query, keys)
        elif self.method == 'concat':
            attn_energies = self.concat_score(query, keys)
        elif self.method == 'dot':
            attn_energies = self.dot_score(query, keys)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1)  # dim=(1,keys.dim(0))


class Decoder(nn.Module):
    # embedding_medications: embedding matrix that transform medication code to dense representation vectors
    # hop: #hop that will be conducted on the key-value memory network
    # attn_type: attention method, including dot, general, and concat
    def __init__(self, device, hidden_size, output_size, medications_count, hop, embedding_medications_np=None,
                 dropout_rate=0, attn_type_kv='dot', attn_type_embedding='dot', ehr_adj=None):
        super(Decoder, self).__init__()

    def forward(self, query, memory_keys, memory_values):
        pass

    def initialize_weights(self):
        pass


class DecoderMultiHopKeyValueSeparate(Decoder):
    def __init__(self, device, hidden_size, output_size, medication_count, hop, embedding_medications_np=None,
                 dropout_rate=0, attn_type_kv='dot', attn_type_embedding='dot', ehr_adj=None):
        super(DecoderMultiHopKeyValueSeparate, self).__init__(device, hop, hidden_size, output_size, medication_count,
                                                              embedding_medications_np, dropout_rate, attn_type_kv,
                                                              attn_type_embedding, ehr_adj)
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_medications = nn.Embedding(medication_count, hidden_size)  # dim=(#medication, hidden_size)
        self.medication_count = medication_count
        self.hop_count = hop
        self.dropout_rate = dropout_rate
        self.attn_type_kv = attn_type_kv
        self.attn_type_embedding = attn_type_embedding

        self.dropout = nn.Dropout(self.dropout_rate)
        self.attn_kv = Attn(self.attn_type_kv, hidden_size)
        self.attn_embedding = Attn(self.attn_type_embedding, hidden_size)
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size * 3, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, output_size))
        self.initialize_weights()

        if embedding_medications_np is not None:  # use pretrained embedding vectors to initialize the embeddings
            print('use pretrained embedding vectors to initialize medication embeddings')
            self.embedding_medications.weight.data.copy_(torch.from_numpy(embedding_medications_np))

    def forward(self, query, memory_keys, memory_values):
        if memory_keys is None:  # no historical admission, memory_keys=None, memory_values=None
            # tmp_indices = torch.from_numpy(np.arange(0, self.medication_count)).to(self.device)
            tmp_indices = torch.LongTensor(np.arange(0, self.medication_count)).to(self.device)
            embedded_medications = self.embedding_medications(tmp_indices)
            weights_embedding = self.attn_embedding(query, embedded_medications)  # dim=(1,#medication)
            context_e = torch.mm(weights_embedding, self.dropout(embedded_medications))  # dim=(1,hidden_size)
            context_o = context_e

        else:
            memory_values_multi_hot = np.zeros((len(memory_values), self.medication_count))
            for idx, admission in enumerate(memory_values):
                memory_values_multi_hot[idx, admission] = 1
            memory_values_multi_hot = torch.FloatTensor(memory_values_multi_hot).to(self.device)
            # tmp_indices = torch.from_numpy(np.arange(0, self.medication_count)).to(self.device)
            tmp_indices = torch.LongTensor(np.arange(0, self.medication_count)).to(self.device)

            weights_kv = self.attn_kv(query, memory_keys)
            weights_medications = torch.mm(weights_kv, memory_values_multi_hot)
            current_o = torch.mm(weights_medications, self.dropout(self.embedding_medications(tmp_indices)))
            context_o = torch.add(query, current_o)
            for hop in range(1, self.hop_count):
                weights_kv = self.attn_kv(context_o, memory_keys)
                weights_medications = torch.mm(weights_kv, memory_values_multi_hot)
                current_o = torch.mm(weights_medications, self.dropout(self.embedding_medications(tmp_indices)))
                context_o = torch.add(context_o, current_o)
            embedded_medications = self.embedding_medications(tmp_indices)
            weights_embedding = self.attn_embedding(query, embedded_medications)  # dim=(1,#medication)
            context_e = torch.mm(weights_embedding, self.dropout(embedded_medications))  # dim=(1,hidden_size)

        output = self.output(torch.cat([query, context_o, context_e], -1))  # dim=(1,output_size)
        return output

    def initialize_weights(self):
        init_range = 0.1
        self.embedding_medications.weight.data.uniform_(-init_range, init_range)
        self.attn_kv.initialize_weights(init_range)
        self.attn_embedding.initialize_weights(init_range)


"""
networks for GAN model
"""


class DiscriminatorMLPPremium(nn.Module):
    def __init__(self, hidden_size, dropout_rate, n_hidden_layers=1, dim_B=128, dim_C=16, use_GPU=False):
        super(DiscriminatorMLPPremium, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.n_hidden_layers = n_hidden_layers
        self.dim_B = dim_B
        self.dim_C = dim_C
        self.use_GPU = use_GPU
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.priori_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.priori_fc2 = nn.Linear(hidden_size * 2, hidden_size * 3)
        self.module_list = nn.ModuleList(
            [nn.Linear(hidden_size * 3, hidden_size * 3) for _ in range(self.n_hidden_layers)])
        self.posterior_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.posterior_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size + self.dim_B, 1)  # dim_B for minibatch discrimination
        self.sigmoid = nn.Sigmoid()

        # minibatch discrimination
        T_ten_init = torch.randn(hidden_size, self.dim_B * self.dim_C)
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)

        self.initialize_weights()

    def forward(self, input_data):
        x = F.relu(self.priori_fc1(input_data))
        x = F.relu(self.priori_fc2(x))
        for layer in self.module_list:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = F.relu(self.posterior_fc1(x))
        feature = F.relu(self.posterior_fc2(x))  # for feature matching, dim=(batch_size,hidden_size)

        # minibatch discrimination
        T_tensor = self.T_tensor
        if self.use_GPU:
            T_tensor = T_tensor.cuda()
        Ms = feature.mm(T_tensor)  # dim=(batch_size,dim_B*dim_C)
        Ms = Ms.view(-1, self.dim_B, self.dim_C)  # dim=(batch_size,dim_B,dim_C)

        outer_tensor = []
        for i in range(Ms.size()[0]):  # for each sample
            out_i = None
            for j in range(Ms.size()[0]):  # interact with all the other samples
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i
            outer_tensor.append(out_i)
        out_T = torch.cat(tuple(outer_tensor)).view(Ms.size()[0], self.dim_B)  # dim=(batch_size,dim_B)
        x = torch.cat((feature, out_T), 1)  # dim=(batch-size,hidden_size+dim_B)
        # minibatch discrimination finished

        x = self.output(x)
        x = self.sigmoid(x)
        return feature, x

    def initialize_weights(self):
        init_range = 0.1
        self.priori_fc1.weight.data.uniform_(-init_range, init_range)
        self.priori_fc1.bias.data.zero_()
        self.priori_fc2.weight.data.uniform_(-init_range, init_range)
        self.priori_fc2.bias.data.zero_()
        for layer in self.module_list:
            layer.weight.data.uniform_(-init_range, init_range)
            layer.bias.data.zero_()
        self.posterior_fc1.weight.data.uniform_(-init_range, init_range)
        self.posterior_fc1.bias.data.zero_()
        self.posterior_fc2.weight.data.uniform_(-init_range, init_range)
        self.posterior_fc2.bias.data.zero_()
        self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.zero_()


class DiscriminatorMLPPremiumForTuning(nn.Module):
    def __init__(self, hidden_size, dropout_rate, n_hidden_layers=1, dim_B=128, dim_C=16, use_GPU=False):
        super(DiscriminatorMLPPremiumForTuning, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.n_hidden_layers = n_hidden_layers
        self.dim_B = dim_B
        self.dim_C = dim_C
        self.use_GPU = use_GPU
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.priori_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.priori_fc2 = nn.Linear(hidden_size * 2, hidden_size * 3)
        self.module_list = nn.ModuleList(
            [nn.Linear(hidden_size * 3, hidden_size * 3) for _ in range(self.n_hidden_layers)])
        self.posterior_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.posterior_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size + self.dim_B, 1)  # dim_B for minibatch discrimination
        # self.output = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # minibatch discrimination
        T_ten_init = torch.randn(hidden_size, self.dim_B * self.dim_C)
        self.T_tensor = nn.Parameter(T_ten_init, requires_grad=True)

        self.initialize_weights()

    def forward(self, input_data):
        x = F.relu(self.priori_fc1(input_data))
        x = F.relu(self.priori_fc2(x))
        for layer in self.module_list:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = F.relu(self.posterior_fc1(x))
        feature = F.relu(self.posterior_fc2(x))  # for feature matching, dim=(batch_size,hidden_size)

        # # minibatch discrimination
        T_tensor = self.T_tensor
        if self.use_GPU:
            T_tensor = T_tensor.cuda()
        Ms = feature.mm(T_tensor)  # dim=(batch_size,dim_B*dim_C)
        Ms = Ms.view(-1, self.dim_B, self.dim_C)  # dim=(batch_size,dim_B,dim_C)

        outer_tensor = []
        for i in range(Ms.size()[0]):  # for each sample
            out_i = None
            for j in range(Ms.size()[0]):  # interact with all the other samples
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i
            outer_tensor.append(out_i)
        out_T = torch.cat(tuple(outer_tensor)).view(Ms.size()[0], self.dim_B)  # dim=(batch_size,dim_B)
        x = torch.cat((feature, out_T), 1)  # dim=(batch_size,hidden_size+dim_B)
        # # minibatch discrimination finished

        x = self.output(x)
        x = self.sigmoid(x).view(-1)  # for hyperparameters tuning
        # x = self.sigmoid(x)  # for discriminator training
        return x

    def initialize_weights(self):
        init_range = 0.1
        self.priori_fc1.weight.data.uniform_(-init_range, init_range)
        self.priori_fc1.bias.data.zero_()
        self.priori_fc2.weight.data.uniform_(-init_range, init_range)
        self.priori_fc2.bias.data.zero_()
        for layer in self.module_list:
            layer.weight.data.uniform_(-init_range, init_range)
            layer.bias.data.zero_()
        self.posterior_fc1.weight.data.uniform_(-init_range, init_range)
        self.posterior_fc1.bias.data.zero_()
        self.posterior_fc2.weight.data.uniform_(-init_range, init_range)
        self.posterior_fc2.bias.data.zero_()
        self.output.weight.data.uniform_(-init_range, init_range)
        self.output.bias.data.zero_()
