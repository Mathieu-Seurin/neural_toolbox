import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
from torch.autograd import Variable
from neural_toolbox.gpu_utils import FloatTensor

from .fusion_utils import TextAttention, choose_reduction_method


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (1 + gammas) * x + betas
        #return gammas * x + betas

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):
        super(ResidualBlock, self).__init__()

        self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result
        self.conv_main = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # Adding coord maps
        w, h, batch_size = x.size(2), x.size(3), x.size(0)
        coord = coord_map((w,h)).expand(batch_size, -1, -1, -1).type_as(x)
        x = torch.cat([x, coord], 1)

        #Before residual connection
        after_proj = self.conv_proj(x)
        after_proj = F.relu(after_proj)

        # Second convolution, relu batch norm
        output = self.conv_main(after_proj)
        output = self.bn(output)
        output = F.relu(output)

        # Add residual connection
        output = self.residual(after_proj,output)

        return output

class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, with_residual, with_batchnorm):

        super(FiLMedResBlock, self).__init__()

        self.conv_proj = nn.Conv2d(in_dim + 2, out_dim, kernel_size=1, stride=1)

        if with_batchnorm:
            self.bn = nn.BatchNorm2d(out_dim, affine=False)
        else:
            self.bn = lambda x:x

        if with_residual:
            self.residual = lambda x,result : x+result
        else:
            self.residual = lambda x,result: result

        self.conv_main = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.film = FiLM()

    def forward(self, x, gammas, betas):

        # Adding coord maps
        w, h, batch_size = x.size(2), x.size(3), x.size(0)
        coord = coord_map((w,h)).expand(batch_size, -1, -1, -1).type_as(x)
        x = torch.cat([x, coord], 1)

        #Before residual connection
        after_proj = self.conv_proj(x)
        after_proj = F.relu(after_proj)

        # Conv before film
        output = self.conv_main(after_proj)
        output = self.bn(output)

        # Film
        output = self.film(output, gammas, betas)
        output = F.relu(output)

        # Add residual connection
        output = self.residual(after_proj,output)

        return output


class SimpleFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, input_size):
        super(SimpleFilmGen, self).__init__()

        self.n_block_to_modulate = n_block_to_modulate

        self.n_feature_map_per_block = n_feature_map_per_block
        self.n_features_to_modulate = self.n_block_to_modulate * self.n_feature_map_per_block

        self.output_mlp = self.n_features_to_modulate * 2
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated


        #self.film_gen_hidden = nn.Linear(input_size, self.film_gen_hidden_size)
        self.film_gen_last_layer = nn.Linear(input_size, self.output_mlp)

        #self.bn_output = nn.BatchNorm1d(self.output_mlp, affine=False)

    def forward(self, text, first_layer, visual_integration=None, init_with_vision=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        if first_layer:
            self.num_layer_count = 0
            #hidden_film_gen_activ = F.relu(self.film_gen_hidden(text))
            #self.gammas_betas = self.bn_output(self.film_gen_last_layer(text))
            self.gammas_betas = self.film_gen_last_layer(text)

        gamma_beta_id = slice(self.n_feature_map_per_block * self.num_layer_count * 2,
                              self.n_feature_map_per_block * (self.num_layer_count + 1) * 2)

        self.num_layer_count += 1

        return self.gammas_betas[:, gamma_beta_id]

class MultiHopFilmGen(nn.Module):

    def __init__(self, config, n_block_to_modulate, n_feature_map_per_block, text_size, vision_size=None, vision_extractor_size_output=None):
        super(MultiHopFilmGen, self).__init__()
        assert vision_size is not None, "For FiLM with feedback loop, need size of visual features"

        self.text_size = text_size
        self.vision_size = vision_size
        self.use_feedback = config["use_feedback"]

        self.n_feature_map_per_block = n_feature_map_per_block

        self.attention = TextAttention(hidden_mlp_size=config["attention_size_hidden"],
                                       text_size=text_size)

        vision_reducing_method = config["vision_reducing_method"]
        vision_reducing_size_mlp = config["vision_reducing_size_mlp"]

        if vision_extractor_size_output:

            batch_size = vision_extractor_size_output[0]
            in_features = vision_extractor_size_output[1]
            width = vision_extractor_size_output[2]

            self._init_ht_conv = nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1)
            self._init_ht_pooling = nn.MaxPool2d(kernel_size=4)
            in_features = self._init_ht_pooling(self._init_ht_conv(Variable(torch.zeros(*vision_extractor_size_output), volatile=True)))
            in_features = in_features.view(batch_size, -1).size(1)

            self._init_ht_mlp = nn.Linear(in_features, self.text_size)

            self.init_ht = self.compute_reduction_init

        else:
            self.init_ht = lambda batch : Variable(torch.ones(batch.size(0), self.text_size).type(FloatTensor))

        if self.use_feedback:
            self.vision_reducer_layer = choose_reduction_method(vision_reducing_method,
                                                                vision_extractor_size_output,
                                                                vision_reducing_size_mlp=vision_reducing_size_mlp)
        else:
            vision_after_reduce_size = 0

        #self.film_gen_hidden = nn.Linear(self.text_size + vision_after_reduce_size, self.film_gen_hidden_size)

        self.film_gen_last_layer = nn.Linear(self.text_size + vision_after_reduce_size, n_feature_map_per_block * 2)
        # for every feature_map, you generate a beta and a gamma, to do : feature_map*gamma + beta
        # So, for every feature_map, 2 parameters are generated

    def forward(self, text, first_layer, visual_integration=None):
        """
        Common interface for all Film Generator
        first_layer indicate that you calling film generator for the first time (needed for init etc...)
        """
        # todo : learn init from vision ??
        # if first layer, reset ht to ones only or init with vision, maybe better.
        if first_layer:
            self.ht = self.init_ht(visual_integration)

        # Compute text features
        text_vec = self.attention(text_seq=text, previous_hidden=self.ht)
        # todo layer norm ? not available on 0.3.0
        self.ht = text_vec


        # Compute feedback loop and fuse
        if self.use_feedback:
            vision_feat_reduced = self.vision_reducer_layer(visual_integration)
            film_gen_input = torch.cat((vision_feat_reduced, text_vec), dim=1)
        else:
            film_gen_input = text_vec

        # Generate film parameters
        gammas_betas = self.film_gen_last_layer(film_gen_input)

        return gammas_betas

    def compute_reduction_init(self, vision_extractor_features):

        # Detach the variable from graph, to avoid changing the features extractor.
        init_ht = vision_extractor_features.data

        init_ht = Variable(init_ht.type(FloatTensor))
        init_ht = self._init_ht_pooling(F.relu(self._init_ht_conv(init_ht)))

        init_ht = init_ht.view(init_ht.size(0), -1)
        init_ht = self._init_ht_mlp(init_ht)

        return init_ht

def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = kaiming_normal
    elif init.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)
        elif isinstance(m, nn.GRU):
            init_params(m.weight_hh_l0)
            init_params(m.weight_ih_l0)


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n)
    y_coord_row = torch.linspace(start, end, steps=m)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))