import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

class FeatureReweightingModule(nn.Module):

    def __init__(self, input_features, output_dim=4, num_layers=1, features=96):
        super(FeatureReweightingModule, self).__init__()
        self.input_features = input_features
        self.features = features
        self.output_dim = output_dim
        self.num_layers = num_layers

        for i in range(self.num_layers):
            module_adjacency = AdjacencyMatrixPrediction(input_features=self.input_features + int(self.features / 2) * i,
                features = features, operator = 'J2', activation = 'softmax', ratio = [2, 2, 1, 1])
            graph = GraphModule(features_input = self.input_features + int(features / 2) * i,
                features_output = int(features / 2), J = 2)
            self.add_module(f'adjacency_{i}', module_adjacency)
            self.add_module(f'graph_{i}', graph)

        self.last_adjacency = AdjacencyMatrixPrediction(input_features=self.input_features + int(self.features / 2) * self.num_layers,
            features = features, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = GraphModule(features_input = self.input_features + int(self.features / 2) * self.num_layers,
            features_output = self.output_dim, J = 2, bn_bool=False)

    def forward(self, x):
        identity = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)) # Identity per batch element, the start of Graph Neural Network
        identity = identity.to(x.device)
        classifiers = []
        for i in range(self.num_layers):
            adjacency, classifier = self._modules['adjacency_{}'.format(i)](x, identity)#calculate adjancent matrix
            classifiers.append(classifier)

            x_new = self._modules[f'graph_{i}'](adjacency, x)
            x_new = x_new[1]
            x_new = F.leaky_relu(x_new)

            x = torch.cat([x, x_new], 2)

        adjacency, classifier = self.last_adjacency(x, identity)
        classifiers.append(classifier)
        x_new = self.layer_last(adjacency, x)[1]

        return x_new[:, :, :], classifiers



def gmul(adjacency, x):
    # x is a tensor of size (bs, N, num_features). [40, 26, 181]
    # W is a tensor of size (bs, N, N, J), [40, 26, 26, 2]
    adjacency_size = adjacency.size()
    N = adjacency_size[-2]
    adjacency = adjacency.split(1, 3)#split in dim=3, every part is 1
    adjacency = torch.cat(adjacency, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)   x is a tensor of size (bs, N, num_features)
    output = torch.bmm(adjacency, x) # output has size (bs, J*N, num_features), https://pytorch.org/docs/stable/torch.html#torch.bmm
    output = output.split(N, 1) #spli into  a list with length=J, size: [bs, N, num_features]
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class GraphModule(nn.Module):
    def __init__(self, features_input, features_output, J, bn_bool=False):
        super(GraphModule, self).__init__()
        self.J = J
        self.num_inputs = J * features_input
        self.num_outputs = features_output
        self.fully_connected = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, adjacency, x):
        x = gmul(adjacency, x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fully_connected(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(x_size[0], x_size[1], self.num_outputs)
        return adjacency, x


class AdjacencyMatrixPrediction(nn.Module):

    def __init__(self, input_features, features, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=True, bn=False):
        super(AdjacencyMatrixPrediction, self).__init__()
        self.features = features
        self.operator = operator
        self.bn = bn
        self.drop = drop
        self.convolutions = []
        self.input_features = input_features
        current_input_features = self.input_features
        current_output_features = int(self.features * ratio[0])
        for layer in range(4):
            current_layers = []
            current_layers.append(nn.Conv2d(current_input_features, current_output_features, 1, stride=1))
            if self.bn:
                current_layers.append(nn.BatchNorm2d(current_output_features))
            current_layers.append(nn.LeakyReLU())
            if self.drop and layer == 1:
                current_layers.append(nn.Dropout(0.3))
            self.convolutions.append(nn.Sequential(*current_layers))
            current_input_features = current_output_features
            current_output_features = int(self.features * ratio[layer])
        self.convolutions = nn.Sequential(*self.convolutions)

        self.conv2d_last = nn.Conv2d(features, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, support, identity_adjacency):
        support = support.unsqueeze(2)
        support_transposed = torch.transpose(support, 1, 2)
        cross_difference = torch.abs(support - support_transposed)
        cross_difference = torch.transpose(cross_difference, 1, 3)

        cross_difference = self.convolutions(cross_difference)

        cross_difference = self.conv2d_last(cross_difference)
        cross_difference = torch.transpose(cross_difference, 1, 3)

        #if self.activation == 'softmax':

        cross_difference = cross_difference - ( identity_adjacency.expand_as(cross_difference) * 1e8 )
        cross_difference = torch.transpose(cross_difference, 2, 3) # after transpose, [bs, N, 1, N]
        # Applying Softmax
        cross_difference = cross_difference.contiguous()
        cross_difference_size = cross_difference.size()
        cross_difference = cross_difference.view(-1, cross_difference.size(3))
        cross_difference = F.softmax(cross_difference)
        cross_difference = cross_difference.view(cross_difference_size)
        # Softmax applied
        cross_difference = torch.transpose(cross_difference, 2, 3)
        '''
        elif self.activation == 'sigmoid':
            cross_difference = F.sigmoid(cross_difference)
            cross_difference *= (1 - identity_adjacency)
        elif self.activation == 'none':
            cross_difference *= (1 - identity_adjacency)
        else:
            raise (NotImplementedError)
        '''


        if self.operator == 'laplace':
            output = identity_adjacency - cross_difference
        elif self.operator == 'J2':
            output = torch.cat([identity_adjacency, cross_difference], 3)
        else:
            raise(NotImplementedError)

        return output, cross_difference



class LayerNorm(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps)


class SpatialSimilarityModule(nn.Module):

    def __init__(self, in_channels=3, inter_channels=None, sub_sample=True, bn_layer=True, dropout= 0.2):
        super(SpatialSimilarityModule, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.query_convolution_classifier = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.support_convolution_classifier = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = LayerNorm()

        self.support_convolution = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if sub_sample:
          self.support_convolution = nn.Sequential(self.support_convolution, nn.MaxPool2d(kernel_size=(2, 2)))
          self.support_convolution_classifier = nn.Sequential(self.support_convolution_classifier, nn.MaxPool2d(kernel_size=(2, 2)))

        self.combined_convolution = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.combined_convolution.weight, 0)
        nn.init.constant(self.combined_convolution.bias, 0)
        if bn_layer:
          self.combined_convolution = nn.Sequential(self.combined_convolution, nn.BatchNorm2d(self.in_channels))


    def forward(self, query, support):
        # BATCH, CHANNELS, WIDTH, HEIGHT
        batch_size = query.size(0)
        #print(query.size(), support.size()) # OK

        # coordinate classifier (coordinate matcher)
        query_transformed_classifier = self.query_convolution_classifier(query).view(batch_size, self.inter_channels, -1)
        query_transformed_classifier = query_transformed_classifier.permute(0, 2, 1) # BATCH, WIDTH x HEIGHT, CHANNELS

        support_transformed_classifier = self.support_convolution_classifier(support).view(batch_size, self.inter_channels, -1)
        # BATCH, CHANNELS, (WIDTH / 2) x (HEIGHT / 2)

        classifier = torch.matmul(query_transformed_classifier, support_transformed_classifier)
        classifier = classifier / math.sqrt(self.inter_channels)
        classifier = F.softmax(classifier, dim=-1)
        #print(classifier.size(), classifier.sum(2))

        # filter support using classifier
        support_transformed = self.support_convolution(support).view(batch_size, self.inter_channels, -1)
        support_transformed = support_transformed.permute(0, 2, 1)
        support_filtered = torch.matmul(classifier, support_transformed)
        support_filtered = self.layer_normalization(support_filtered)
        #print(support_filtered.size())
        support_filtered = support_filtered.permute(0, 2, 1).contiguous()
        support_filtered = support_filtered.view(batch_size, self.inter_channels, *query.size()[2:])
        support_filtered = self.dropout(self.combined_convolution(F.relu(support_filtered)))

        support_query_combined = support_filtered + query

        return support_query_combined, classifier, support_filtered

class Silco(nn.Module):

    def __init__(self):
        super(Silco, self).__init__()

        self.feature_reweigting = \
            nn.ModuleList([FeatureReweightingModule(input_features=24, output_dim=1),
             FeatureReweightingModule(input_features=32, output_dim=1),
             FeatureReweightingModule(input_features=64, output_dim=1),
             FeatureReweightingModule(input_features=160, output_dim=1)])

        self.spatial_similarity = \
            nn.ModuleList([SpatialSimilarityModule(in_channels=24, inter_channels=24),
                           SpatialSimilarityModule(in_channels=32, inter_channels=32),
                           SpatialSimilarityModule(in_channels=64, inter_channels=64),
                           SpatialSimilarityModule(in_channels=160, inter_channels=160) ])

        self.relu = nn.ReLU(inplace=True)

        self.conv_fusions = nn.ModuleList([nn.Conv2d(24, 24, 1, stride=1),
                                          nn.Conv2d(32, 32, 1, stride=1),
                                          nn.Conv2d(64, 64, 1, stride=1),
                                          nn.Conv2d(160, 160, 1, stride=1)])


    def fuse(self, feat_query, support_features, context_level):
        batch_size = feat_query.size()[0]
        channels = feat_query.size()[1]
        width = feat_query.size()[2]
        height = feat_query.size()[3]
        batch_size_support = support_features.size()[0]
        support_count = int(batch_size_support / batch_size)
        original_support  = support_features  # [B*S,C,W,H]
        global_averge_pooling = nn.AvgPool2d(support_features.size()[2:])  # BSxCx1x1

        # channel attention feature
        support_unfolded = original_support.view(batch_size, support_count, channels, width, height)

        transformed_supports = []
        spatial_classifiers = []
        spatial_supports = []
        for support_index in range(support_count):
            current_support, classifier, support_filtered = \
                self.spatial_similarity[context_level](feat_query, support_unfolded[:, support_index, :, :, :])
            current_support = current_support.\
                view(batch_size, channels, width, height)
            transformed_supports.append(current_support.view(batch_size, 1, channels, width, height))
            spatial_classifiers.append(classifier)
            spatial_supports.append(support_filtered)
        transformed_supports = torch.cat(transformed_supports, 1)

        support_global_average = torch.squeeze(global_averge_pooling(support_features.view(-1, channels, width, height)))  # BSxC
        support_global_average = support_global_average.view(-1, support_count, channels)  # BxSxC

        support_reweighted, feature_reweighting_classifier = self.feature_reweigting[context_level](support_global_average)
        support_reweighted = F.sigmoid(support_reweighted.view(-1, support_count, 1, 1, 1))  # [B, S, 1]=>[B, S, C, 1, 1]

        transformed_supports = torch.sum(transformed_supports * support_reweighted, dim=1, keepdim=False)
        return self.relu(self.conv_fusions[context_level](transformed_supports)), spatial_classifiers, feature_reweighting_classifier, spatial_supports


    def forward(self, queries, supports):
        updated_queries = []
        query_spatial_classifiers = []
        query_feature_classifier = []
        query_spatial_supports = []
        for scale_index, query_at_scale in enumerate(queries):
            updated_query, spatial_classifiers, feature_classifier, spatial_supports = self.fuse(query_at_scale, supports[scale_index], context_level=scale_index)
            updated_queries.append(updated_query)
            query_spatial_classifiers.append(spatial_classifiers)
            query_feature_classifier.append(feature_classifier)
            query_spatial_supports.append(spatial_supports)
        return updated_queries, query_spatial_classifiers, query_feature_classifier, query_spatial_supports

    @staticmethod
    def backbone_features(backbone, queries, supports):
        support_features_compacted = None
        if supports is not None:
            supports_compacted = supports.view(-1, *queries.size()[1:])
            support_features_compacted = backbone(supports_compacted)
        query_features = backbone(queries)
        return query_features, support_features_compacted


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    '''
    # spatial similarity module
    query = Variable(torch.zeros(2, 3, 4, 4))
    support = Variable(torch.zeros(2, 3, 4, 4))
    net = SpatialSimilarityModule()
    out = net(query, support)
    print(out.size())


    # feature reweighting module
    input_features = 3 * 10 * 10
    output_dim = 4
    num_layers = 1
    feature_reweighting_module = FeatureReweightingModule(input_features, output_dim, num_layers, features=96).cuda()
    batch_size = 2
    samples = 5
    input = torch.FloatTensor(batch_size, samples, input_features).cuda()
    output = feature_reweighting_module(input)
    print(input.size(), output.size())
    '''


    from roboaugen.core.config import Config
    from roboaugen.model.models import MobileNetV2, HigherResolutionNetwork

    def backbone_features(backbone, queries, supports):
        supports_compacted = supports.view(-1, *queries.size()[1:])
        support_features_compacted = backbone(supports_compacted)
        '''
        support_features = []
        for support_feature_compacted in support_features_compacted:
            support_feature = support_feature_compacted.view(*supports.size()[0:2], *support_feature_compacted.size()[1:])
            support_features.append(support_feature)
        '''
        query_features = backbone(queries)
        return query_features, support_features_compacted

    batch_size = 2
    support_count = 5
    color_channels = 3
    width  = 200
    height = 200
    config = Config()

    # backbone
    queries = torch.FloatTensor(batch_size, color_channels, width, height).cuda()
    supports = torch.FloatTensor(batch_size, support_count, color_channels, width, height).cuda()
    backbone = MobileNetV2(3).cuda()
    query_features, support_features = backbone_features(backbone, queries, supports)

    # silco
    silco = Silco().cuda()
    updated_query_features = silco(query_features, support_features)

    # high resolution network
    higher_resolution_network = HigherResolutionNetwork(config.channels_blocks, config.dimensions_blocks, config.num_vertices).cuda()
    heatmaps = higher_resolution_network(updated_query_features)



