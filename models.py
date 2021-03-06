import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from neural_toolbox.film_utils import ResidualBlock, FiLMedResBlock, init_modules, MultiHopFilmGen, SimpleFilmGen, Mlp_Modulation
from neural_toolbox.gpu_utils import FloatTensor, USE_CUDA, LongTensor

from torch.nn import CrossEntropyLoss
import logging
import numpy as np


def count_good_prediction(yhat,y):
    model_pred = torch.max(yhat, 1)[1].cpu().data.numpy()
    y = y.cpu().data.numpy()
    return np.equal(model_pred,y).sum()

def compute_accuracy(yhat, y):
    return count_good_prediction(yhat=yhat,y=y) / len(yhat)


class ClfModel(nn.Module):
    def __init__(self, config, n_class, input_info):

        super(ClfModel, self).__init__()

        self.forward_model = FilmedNet(config=config,
                                       n_class=n_class,
                                       input_info=input_info)

        # Init Loss
        self.loss_func = CrossEntropyLoss()

        if USE_CUDA :
            self.forward_model.cuda()
            self.loss_func.cuda()

        if config["weights_to_load"]:
            self.load_state_dict(torch.load(config["weights_to_load"]), strict=False)
            # To avoid batch norm problem, set training to false
            self.forward_model.train(mode=False)
        else:
            init_modules(self.modules())

    def forward(self, x):
        return self.forward_model(x)

    def optimize(self, x, y):

        yhat = self.forward_model(x)
        loss = self.loss_func(yhat, y)

        self.forward_model.optimizer.zero_grad()
        loss.backward()

        # for param in self.forward_model.parameters():
        #     #logging.debug(param.grad.data.sum())
        #     param.grad.data.clamp_(-1., 1.)

        self.forward_model.optimizer.step()

        return loss.data.cpu(), yhat

    def new_task(self, num_task):

        raise NotImplementedError("Not available yet")

        if num_task == 0:
            self.forward_model.use_film = False

        # todo : change hardcoded, do scheduler of something like this ?
        if num_task > 0:

            # todo : check resnet
            self.forward_model.use_film = True
            for param_group in self.forward_model.optimizer.param_groups:
                if param_group['name'] == "base_net_params":
                    param_group['lr'] = self.forward_model.after_lr
                elif param_group['name'] == "film_params":
                    param_group['lr'] = self.forward_model.default_lr


class TextEmbedEncoder(nn.Module):
    def __init__(self,config, need_all_ht, text_size, vocab_size):
        super(TextEmbedEncoder, self).__init__()

        # Dealing with text / second input
        embedding_dim = config["embedding_dim"]
        self.rnn_hidden_size = config["rnn_hidden_size"]
        self.text_size = text_size

        # simple film need just the last state, multi-hop need all ht.
        self.return_all_ht = need_all_ht

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim
                                      )

        self.rnn = nn.GRU(num_layers=config["n_rnn_layers"],
                          input_size=embedding_dim,
                          hidden_size=self.rnn_hidden_size,
                          batch_first=True,
                          bidirectional=False)

    def forward(self, text):

        batch_size = text.size(0)

        ids_last_word = self.retrieve_last_index(text)

        embedded_q = self.embedding(text)

        h0 = Variable(torch.ones(1, batch_size, self.rnn_hidden_size).type(FloatTensor))
        all_ht, _ = self.rnn(embedded_q, h0)

        if self.return_all_ht:
            return all_ht
        else:
            return self.get_last_ht_batch(all_ht, ids_last_word)

    def retrieve_last_index(self, text):
        """
        All sentences are padded to match the size of the largest sentence.
        This function just retrieve the index of the first padding token in every sentence
        """
        text = text.data.cpu()
        last_indexes = []

        for sent in text:
            index = sent.nonzero()[-1,0]
            last_indexes.append(index)

        return last_indexes

    def get_last_ht_batch(self, text_batch, ids):
        """
        To help lstm/gru, instead of retrieving the very last ht of the padded sequence,
        retrieve the last ht in the non-padded sequence.
        """
        batch_size, seq_l, ht_size = text_batch.size()

        ids = Variable(LongTensor(ids)).view(batch_size,1,1).expand(-1,-1,ht_size)
        ht = text_batch.gather(1, ids).squeeze(1)

        return ht


class FilmedNet(nn.Module):
    def __init__(self, config, n_class, input_info):
        super(FilmedNet, self).__init__()

        # General params
        self.default_lr = config["lr"]

        # Resblock params
        self.n_regular_block = config["n_regular_block"]
        self.n_modulated_block = config["n_modulated_block"]
        self.input_shape = list(input_info['vision_shape'])
        self.input_shape[0] = 2
        self.n_channel_input = self.input_shape[1]
        self.n_feature_map_per_block = config["n_feat_map_max"]

        self.regular_blocks = nn.ModuleList()
        self.modulated_blocks = nn.ModuleList()

        # Head params
        self.n_channel_head = config["head_channel"]
        self.kernel_size_head = config["head_kernel"]

        # If use attention as fusing : no head/pooling
        self.use_attention_as_fusing = config['fusing_method'] == 'attention'

        # Film type
        self.use_multihop = False
        self.use_film = config["use_film"]
        if self.use_film:
            self.film_gen_type = config["film_gen_params"]["film_type"]
            self.use_multihop = config["film_gen_params"]["film_type"] == "multi_hop"

        # FC params
        self.fc_n_hidden = config['fc_n_hidden']
        self.fc_dropout = config["fc_dropout"]
        self.n_class = n_class

        # Second modality (aka text for clevr)
        self.second_modality_shape = input_info['second_modality_shape']

        if input_info["second_modality_type"] == "text":
            self.text_embed_encode = TextEmbedEncoder(config['text_encoding'],
                                                      need_all_ht=self.use_multihop,
                                                      text_size=self.second_modality_shape,
                                                      vocab_size=input_info['vocab_size'])
            # todo : deal with multiple ht
            self.encoded_text_size = self.text_embed_encode.rnn_hidden_size
        else:
            task_embedding_size = config["task_encoding"]["embedding_dim"]
            self.text_embed_encode = nn.Linear(1, task_embedding_size)
            self.encoded_text_size = task_embedding_size


        #== Learned features extractor ===
        #=================================
        self.n_feature_extactor_channel = config["features_extractor"]["n_channel"]
        self.feature_extactor_kernel = config["features_extractor"]["kernel_size"]
        self.feature_extactor_stride = config["features_extractor"]["stride_size"]


        self.feature_extactor = nn.Sequential()
        n_channel_to_next_block = self.n_channel_input

        for layer in range(config["features_extractor"]["n_layer"]):
            self.feature_extactor.add_module("conv"+str(layer), nn.Conv2d(in_channels=n_channel_to_next_block,
                                                                          out_channels=self.n_feature_extactor_channel,
                                                                          kernel_size=self.feature_extactor_kernel,
                                                                          stride=self.feature_extactor_stride,
                                                                          padding=1))

            self.feature_extactor.add_module("bn"+str(layer), nn.BatchNorm2d(self.n_feature_extactor_channel))
            self.feature_extactor.add_module("relu"+str(layer), nn.ReLU())

            n_channel_to_next_block = self.n_feature_extactor_channel


        # RESBLOCK BUILDING
        #==================
        # Create resblock, not modulated by FiLM
        for regular_resblock_num in range(self.n_regular_block):

            current_regular_resblock = ResidualBlock(in_dim=n_channel_to_next_block,
                                                     out_dim=self.n_feature_map_per_block,
                                                     with_residual=True,
                                                     with_batchnorm=False)
            n_channel_to_next_block = self.n_feature_map_per_block

            self.regular_blocks.append(current_regular_resblock)



        # Create FiLM-ed resblock
        for modulated_block_num in range(self.n_modulated_block):
            current_modulated_resblock = FiLMedResBlock(in_dim=n_channel_to_next_block,
                                                        out_dim=self.n_feature_map_per_block,
                                                        with_residual=True,
                                                        with_batchnorm=True) #with_cond=[True], dropout=self.resblock_dropout)
            n_channel_to_next_block = self.n_feature_map_per_block

            self.modulated_blocks.append(current_modulated_resblock)


        # head
        if self.kernel_size_head != 0 and not self.use_attention_as_fusing:
            n_channel_head = self.n_class if self.n_channel_head == 'class' else self.n_channel_head
            self.head_conv = nn.Conv2d(in_channels=n_channel_to_next_block,
                                       out_channels=n_channel_head,
                                       kernel_size=self.kernel_size_head)
        else:
            self.head_conv = lambda x:x


        # if head is "class" use it as a classifier, otherwise use MLP (as below)
        if self.n_channel_head != "class" :
            self.use_classif_head = False
            fc_input_size, intermediate_conv_size, extractor_conv_size = self.compute_conv_size()
            self.fc1 = nn.Linear(in_features=fc_input_size, out_features=self.fc_n_hidden)
            self.fc2 = nn.Linear(in_features=self.fc_n_hidden, out_features=self.n_class)
            self.fc_bn = nn.BatchNorm1d(self.fc_n_hidden, affine=False)

        else: # head as classifier (same as inception)
            self.use_classif_head = True

        if self.use_film:

            if self.film_gen_type == "simple":
                self.film_gen = SimpleFilmGen(config=config['film_gen_params'],
                                              n_block_to_modulate=self.n_modulated_block,
                                              n_feature_map_per_block=self.n_feature_map_per_block,
                                              input_size=self.encoded_text_size
                                              )

            elif self.film_gen_type == "multi_hop":
                self.film_gen = MultiHopFilmGen(config=config['film_gen_params'],
                                                n_block_to_modulate=self.n_modulated_block,
                                                n_feature_map_per_block=self.n_feature_map_per_block,
                                                text_size=self.text_embed_encode.rnn_hidden_size,
                                                vision_size=intermediate_conv_size,
                                                vision_extractor_size_output=extractor_conv_size)

            else:
                raise NotImplementedError("Wrong Film generator type : given '{}'".format(self.film_gen_type))


            # Modulator : Before head, after, mlp, stem etc ...
            if config["modulate_top"]["modulate_last_hidden"]:
                self.last_hidden_modulator = Mlp_Modulation(n_in=self.encoded_text_size,
                                                    n_hidden=config["modulate_top"]["n_hidden_last_hidden_layer"],
                                                    n_out=self.fc_n_hidden)
            else:
                self.last_hidden_modulator = lambda x:(0,0)



            if config["modulate_top"]["before_out_modulation"]:

                self.top_modulator = Mlp_Modulation(n_in=self.encoded_text_size,
                                                    n_hidden=config["modulate_top"]["n_hidden_out_layer"],
                                                    n_out=self.n_class)
            else:
                self.top_modulator = lambda x:(0,0)


            if config["modulate_top"]["modulate_stem"]:

                self.stem_modulator = Mlp_Modulation(n_in=self.encoded_text_size,
                                                    n_hidden=config["modulate_top"]["n_hidden_out_layer"],
                                                    n_out=self.n_class)
            else:
                self.stem_modulator = lambda x:(0,0)


            if config["modulate_top"]["modulate_head"]:
                pass
                NotImplementedError("Not yet head modulation, might be useless.")

        # Optimizer
        optimizer = config['optimizer'].lower()

        optim_config = []

        # If you freeze vision, use only modulation part
        if config["freeze_vision"] is True:
            optim_config.append({'params': self.get_modulation_params(), 'lr': config["lr_modulation"]})

        elif config["freeze_vision"] == "partial" :
            optim_config.append({'params': self.get_modulation_params(), 'lr': config["lr_modulation"]})
            optim_config.append({'params': self.get_head_params(), 'lr': config["lr"]})

        elif config["freeze_vision"] == "unfreeze_last2":
            optim_config.append({'params': self.get_modulation_params(), 'lr': config["lr_modulation"]})
            optim_config.append({'params': self.get_head_params(), 'lr': config["lr"]})
            optim_config.append({'params': self.get_resblock_params(list_block_num=[self.n_modulated_block-1,
                                                                                    self.n_modulated_block-2]),
                                 'lr': config["lr"]})




        else:  # If you don't freeze vision, Freeze Film part
            optim_config.append({'params': self.get_all_params_except_modulation(), 'lr': config["lr"]}) # Default config

        # assert len([i for i in optim_config[1]['params']]) + len([i for i in optim_config[0]['params']]) == len([i for i in self.parameters()]), \
        #     "Problem in parameters selection"


        if optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(optim_config, lr=self.default_lr)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(optim_config, lr=self.default_lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(optim_config, lr=self.default_lr)
        else:
            assert False, 'Optimizer not recognized'

    def forward(self, inputs):

        x, second_mod = inputs['vision'], inputs['second_modality']

        second_mod = self.text_embed_encode(second_mod)

        x = self.compute_conv(x, second_mod, still_building_model=False)

        if self.use_classif_head :
            x = F.avg_pool2d(x, kernel_size=x.size(2))
            x = x.view(x.size(0), -1)

            gammas, betas = self.top_modulator(second_mod)
            x = (1+gammas)*x + betas

        else:
            x = x.view(x.size(0), -1)
            x = self.fc_bn(self.fc1(x))
            gammas, betas = self.last_hidden_modulator(second_mod)
            x = (1+gammas)*x + betas

            x = F.relu(x)
            gammas, betas = self.top_modulator(second_mod)
            x = (1+gammas)*x + betas

        return x

    def compute_conv_size(self):

        # Don't convert it to cuda because the model is not yet on GPU (because you're still defining the model here ;)
        tmp = Variable(torch.zeros(*self.input_shape))
        conv_out = self.compute_conv(tmp, still_building_model=True).view(tmp.size(0),-1).size(1)
        return conv_out, self.intermediate_conv_size, self.extactor_size

    def compute_conv(self, x, text_state=None, still_building_model=False):
        """
        :param x: vision input with batch dimension first
        :param text_state: all hidden states of the lstm encoder
        :param still_building_model: needed if you use this function just to get the output size
        :return: return visual features, modulated if FiLM
        """

        if self.use_film:
            if not still_building_model:
                assert text_state is not None, "if you use film, need to provide text as input too"

        batch_size = x.size(0)

        x = self.feature_extactor(x)

        self.extactor_size = x.size()


        # Regular resblock, easy
        for i,regular_resblock in enumerate(self.regular_blocks):
            x = regular_resblock.forward(x)
            self.intermediate_conv_size = x.size()


        #Modulated block : first compute FiLM weights, then send them to the resblock layers
        for i,modulated_resblock in enumerate(self.modulated_blocks):

            if still_building_model or not self.use_film :
                # Gammas = all zeros   Betas = all zeros
                gammas = Variable(torch.zeros(batch_size, self.n_feature_map_per_block).type_as(x.data))
                betas = Variable(torch.zeros_like(gammas.data).type_as(x.data))
            else: # use film
                gammas_betas = self.film_gen.forward(text_state, first_layer= i==0, visual_integration=x)
                assert gammas_betas.size(1)%2 == 0, "Problem, more gammas than betas (or vice versa)"
                middle = gammas_betas.size(1)//2
                gammas = gammas_betas[:,:middle]
                betas = gammas_betas[:, middle:]

            x = modulated_resblock.forward(x, gammas=gammas, betas=betas)
            self.intermediate_conv_size = x.size()

        if not self.use_attention_as_fusing:
            x = self.head_conv(x)
            if not self.use_classif_head:
                x = F.max_pool2d(F.relu(x), kernel_size=x.size()[2:])

        return x

    def get_all_params_except_modulation(self):

        params = []
        for name, param in self.named_parameters():
            if "film_gen" not in name and "modulator" not in name:
                params.append(param)
        return params

    def get_modulation_params(self):

        params = []
        for name, param in self.named_parameters():
            if "film_gen" in name or "modulator" in name:
                params.append(param)
        return params

    def get_head_params(self):

        params = []
        for name, param in self.named_parameters():
            if "head" in name:
                params.append(param)
        return params

    def get_resblock_params(self, list_block_num=None):

        params = []
        list_block_name = ["modulated_blocks."+str(i) for i in list_block_num]

        for name, param in self.named_parameters():
            if any([block_name in name for block_name in list_block_name]):
               params.append(param)
        return params


if __name__ == "__main__":

    yhat = torch.rand(10,4)
    yhat[0:7,3] = 1

    y = torch.ones(10)*2

    print(yhat)
    print(y)
    print(count_good_prediction(yhat=Variable(yhat), y=Variable(y)))

