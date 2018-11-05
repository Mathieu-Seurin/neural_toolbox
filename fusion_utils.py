import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# The mlp has to deal with the image features AND text features
# How do you fuse them ? (concatenation, dot-product, attention)

def choose_fusing(config):
    text_size = config["text_size"]
    vision_size = config["vision_size"]

    if config['fusing_method'] == 'concatenate':
        fuse = concatenate_text_vision
        fusing_output_size = text_size + vision_size

    elif config['fusing_method'] == 'dot':
        embedding_size_before_dot = config['embedding_size_before_dot']
        fuse = DotProduct(embedding_size_before_dot, text_size, vision_size)
        fusing_output_size = embedding_size_before_dot

    elif config['fusing_method'] == 'attention':

        vision_n_feat_map = config["vision_n_featmap"]
        hidden_mlp_size = config["hidden_mlp_size"]
        fuse = VisualAttention(text_size, vision_n_feat_map, hidden_mlp_size)
        # text concatenated with vision after visual_attention, so size = lstm_size + width*heigth
        fusing_output_size = text_size + vision_n_feat_map

    elif config['fusing_method'] == "no_fuse":  # Usual Film method, the text is not used

        # if Film no fuse, the size expected by the fc is the size of visual features, flattened
        fusing_output_size = vision_size
        fuse = vectorize

    else:
        raise NotImplementedError(
            "Wrong Fusion method : {}, can only be : concatenate, dot, attention or no_fuse, but need to be explicit".format(
                config['fusing_method']))

    return fuse, fusing_output_size


def concatenate_text_vision(text, vision):
    vision = vision.view(vision.size(0), -1)
    return torch.cat((vision, text), dim=1)


class DotProduct(nn.Module):
    def __init__(self, embedding_size, text_size, visual_size):
        super(DotProduct, self).__init__()

        self.embedding_size = embedding_size
        self.text_embedding = nn.Linear(text_size, embedding_size)
        self.visual_embedding = nn.Linear(visual_size, embedding_size)

    def forward(self, text, vision):
        vision = vision.view(vision.size(0), -1)

        text = self.text_embedding(text)
        vision = self.visual_embedding(vision)
        return text * vision


class VisualAttention(nn.Module):

    def __init__(self, text_size, vision_n_featmap, hidden_mlp_size):
        super(VisualAttention, self).__init__()

        attention_input_size = text_size + vision_n_featmap
        if hidden_mlp_size > 0:
            hidden_layer_att = nn.Linear(attention_input_size, hidden_mlp_size)
            relu = nn.ReLU()
            self.attention_hidden = nn.Sequential(hidden_layer_att, relu)
        else:
            self.attention_hidden = lambda x: x
            hidden_mlp_size = attention_input_size

        self.attention_last = nn.Linear(hidden_mlp_size, 1)

    def forward(self, text, vision):
        """
        :param text: lstm-encoded text. dim is (batch, hidden_lstm_size)
        :param vision: cnn-encoded image. dim is (batch, n_feature_map, width, height)
        :return: vision after visual attention is applied. dim is (batch, n_feature_map)
        """
        n_feature_map = vision.size(1)
        width = vision.size(2)
        height = vision.size(3)

        attention_weights_list = []
        # compute attention for every pixel, compute the sum
        for i in range(width):
            for j in range(height):
                current_pixel = vision[:, :, i, j]
                assert current_pixel.dim() == 2
                current_weight = self.attention_last(self.attention_hidden(torch.cat((text, current_pixel), dim=1)))
                attention_weights_list.append(current_weight)

        all_weigths = torch.cat(attention_weights_list, dim=1)
        all_weigths = F.softmax(all_weigths, dim=1).unsqueeze(2)

        vision = vision.view(-1, n_feature_map, height * width)
        vision = torch.bmm(vision, all_weigths)
        vision = vision.squeeze(2)

        return concatenate_text_vision(text, vision)


class TextAttention(nn.Module):

    def __init__(self, text_size, hidden_mlp_size):
        super(TextAttention, self).__init__()

        attention_input_size = text_size

        if hidden_mlp_size > 0:
            hidden_layer_att = nn.Linear(attention_input_size, hidden_mlp_size)
            relu = nn.ReLU()
            self.attention_hidden = nn.Sequential(hidden_layer_att, relu)
        else:
            self.attention_hidden = lambda x: x
            hidden_mlp_size = attention_input_size

        self.attention_scoring = nn.Linear(hidden_mlp_size, 1)

    def forward(self, text_seq, previous_hidden):
        """
        :param text_seq: lstm-encoded text. dim is (seq, batch, size_vec)
        :param previous_hidden: last state of the context cell
        :return: text after text_attention is applied. dim is (batch, size_vec)
        """

        sequence_length = text_seq.size(1)
        batch_size = text_seq.size(0)
        size_ht = previous_hidden.size(1)

        attention_weights_list = []

        # compute attention for ht in the sequence
        for ht_num in range(sequence_length):
            current_ht = text_seq[:, ht_num]
            assert current_ht.dim() == 2, "Fail, a single ht should be of dim 2. your dim is {}".format(current_ht.dim())

            dot_product = previous_hidden * current_ht
            current_weight = self.attention_scoring(self.attention_hidden(dot_product))
            attention_weights_list.append(current_weight)

        all_weigths = torch.cat(attention_weights_list, dim=1)
        assert all_weigths.size(1) == sequence_length
        assert all_weigths.size(0) == batch_size

        # all_weigths is [batch_size, weigth], need [batch_size, seq_length, weigth]
        all_weigths = F.softmax(all_weigths, dim=1).unsqueeze(2)

        text_seq = text_seq.permute(0, 2, 1)  # text_seq is dim : (batch, seq_length, ht), need (batch, ht, seq_length)

        text_attentionned = torch.bmm(text_seq, all_weigths)
        text_attentionned = text_attentionned.squeeze(2)

        return text_attentionned


def vectorize(text, vision):
    return vision.view(vision.size(0), -1)


def lstm_last_step(all_ht, last_ht):
    # Warning, works only if you have a 1-layer-monodirectionnal-lstm
    last_ht = last_ht.squeeze(0)
    return last_ht


def lstm_whole_seq(all_ht, last_ht):
    return all_ht.permute(1, 0, 2)


class ConvPoolReducingLayer(nn.Module):
    def __init__(self, in_channel):
        super(ConvPoolReducingLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1)

    def forward(self, vision):
        batch_size = vision.size(0)
        vision = self.conv1(vision)
        vision = F.max_pool2d(vision, kernel_size=3)
        return vision.view(batch_size, -1)


class PoolReducingLayer(nn.Module):
    def __init__(self):
        super(PoolReducingLayer, self).__init__()

    def forward(self, vision):
        batch_size = vision.size(0)
        return F.max_pool2d(vision, kernel_size=4).view(batch_size, -1)


class LinearReducingLayer(nn.Module):
    def __init__(self, vision_size_flatten, output_size):
        super(LinearReducingLayer, self).__init__()
        self.linear = nn.Linear(vision_size_flatten, output_size)

    def forward(self, vision):
        batch_size = vision.size(0)
        vision = vision.view(batch_size, -1)
        vision = self.linear(vision)

        return vision

def choose_reduction_method(vision_reducing_method, vision_size, vision_reducing_size_mlp=None):

    if vision_reducing_method == "mlp":
        assert vision_reducing_size_mlp != None, "Need to specify the size of the vision reducing part in Multi-hop"
        vision_size_flatten = vision_size[1] * vision_size[2] * vision_size[3]  # just flatten the input
        vision_reducer_layer = LinearReducingLayer(vision_size_flatten=vision_size_flatten,
                                                        output_size=vision_reducing_size_mlp)
    elif vision_reducing_method == "conv":
        vision_reducer_layer = ConvPoolReducingLayer(vision_size[1])
    elif vision_reducing_method == "pool":
        vision_reducer_layer = PoolReducingLayer()
    else:
        raise NotImplementedError("Wrong vision reducing method : {}".format(vision_reducing_method))

    tmp = Variable(torch.ones(vision_size), volatile=True)
    tmp_out = vision_reducer_layer(tmp)
    vision_after_reduce_size = tmp_out.size()

    return vision_reducer_layer, vision_after_reduce_size