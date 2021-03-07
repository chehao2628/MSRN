import torch.nn as nn
from torch.autograd import Variable
from util import *

from ResNet import resnet101
from LabelEmbedding import LabelEmbed, GroupEmbed


class MSGDN(nn.Module):
    def __init__(self, num_classes, pool_ratio, model_name, adj_file):
        super(MSGDN, self).__init__()
        self.resnet_101 = resnet101()
        self.adj_file = adj_file
        self.num_classes = num_classes
        self.num_node = int(pool_ratio * num_classes)

        self.label_dim = 512
        self.num_branch = 3

        self.label_embed = LabelEmbed(256)
        self.group_embed = GroupEmbed(self.num_node)

        # build Projection P
        self.ProjectM = nn.ModuleList()
        self.ProjectM.append(nn.Conv2d(512, self.label_dim, 1, 1))
        self.ProjectM.append(nn.Conv2d(1024, self.label_dim, 1, 1))
        self.ProjectM.append(nn.Conv2d(2048, self.label_dim, 1, 1))

        self.fc_label = nn.Linear(300, 256)

        self.fc1 = nn.Linear(self.num_branch * self.label_dim * (self.num_classes + self.num_node), 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)

        self.W = nn.Parameter(torch.Tensor(self.num_classes, self.num_branch * self.label_dim * 2), requires_grad=True)
        self.B = nn.Parameter(torch.Tensor(self.num_classes), requires_grad=True)
        self.reset_parameters()

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, img, inp):
        t = 0.4
        _adj = gen_A(self.num_classes, t, self.adj_file)
        self.A = torch.from_numpy(_adj).float().cuda()
        adj = gen_adj(self.A).detach()
        inp = inp[0]
        inp = torch.as_tensor(inp).cuda()

        # img: image feature; X: label embedding; A: adj matrix
        img_features = self.resnet_101(img)
        # [[8, 256, 56, 56],[8, 1024, 14, 14],[8, 2048, 3, 3]]
        # f1, f2, f3, fx = img_features
        label_embed = self.label_embed(self.fc_label(inp), adj)  # [num_classes, label_dim]
        group_embed, assign_mat = self.group_embed(F.relu(label_embed), adj)  # [4, embedding_length], [20, 4]

        group_predict = assign_mat.argmax(1)
        # group embed loss
        group_loss = torch.Tensor([0]).cuda()
        for i in range(self.num_classes):
            group_loss += torch.dist(label_embed[i], group_embed[group_predict[i]])
        group_loss = 0.0001 * group_loss.squeeze()

        trans_label_feat = label_embed.view(1, 1, 1, self.num_classes, self.label_dim)
        trans_group_feat = group_embed.view(1, 1, 1, self.num_node, self.label_dim)
        label_dr = torch.Tensor().cuda()
        group_dr = torch.Tensor().cuda()
        label_dr = Variable(label_dr, requires_grad=True)
        group_dr = Variable(group_dr, requires_grad=True)

        for s in range(len(img_features)):
            feat = img_features[s]
            self.batch_size, channel, w, h = feat.size()

            feat = self.ProjectM[s](feat)  # [batch_size, dim, w, h]

            # Adjust the dimensions of image feature embedding
            trans_img_feat = torch.transpose(torch.transpose(feat, 1, 2), 2, 3)  # [batch_size, w, h, 256]
            p_feat = trans_img_feat.unsqueeze(3)  # [batch_size, w, h, 1, 256]

            # Cross Modality
            sl, sg = torch.mul(p_feat, trans_label_feat), torch.mul(p_feat,
                                                                    trans_group_feat)  # [batch_size, w, h, 20,30], [batch_size, w, h, 4,30]
            # Normalization
            sl = sl.contiguous().view(self.batch_size, w * h, self.num_classes, self.label_dim)
            sg = sg.contiguous().view(self.batch_size, w * h, self.num_node, self.label_dim)
            nl, ng = F.softmax(sl, dim=1), F.softmax(sg, dim=1)  # [batch_size, w, h, 20,30], [batch_size, w, h, 4,30]

            # Weighted Aggregation
            p_feat = p_feat.contiguous().view(self.batch_size, w * h, 1, self.label_dim)
            agg_l_feat = torch.sum(torch.mul(nl, p_feat), 1)  # [batch_size, dim, 20,1]
            agg_g_feat = torch.sum(torch.mul(ng, p_feat), 1)  # [batch_size, dim, node,1]

            # concat and predict
            label_dr = torch.cat((label_dr, agg_l_feat), -1)  # batch_size x (n*channel) x num_classes
            group_dr = torch.cat((group_dr, agg_g_feat), -1)  # batch_size x (n*channel) x num_node

        ###################################################################################################
        # comment this block and uncomment the next block if applying element wise predict
        # concat and predict
        connected = torch.cat((label_dr, group_dr), 1)  # batch_size x (n*channel) x (num_node+num_classes)
        connected = torch.tanh(connected)
        connected = connected.view(self.batch_size, -1)

        # MLP
        res = F.leaky_relu(self.fc1(connected))
        res = self.fc2(res)
        ###################################################################################################

        ###################################################################################################
        # group_dr = group_dr.permute(0, 2, 1)
        # group_dr = torch.matmul(group_dr, assign_mat.permute(1, 0))
        # group_dr = group_dr.permute(0, 2, 1)
        #
        # connected = torch.cat((label_dr, group_dr), 2)  # batch_size x (n) x (num_node+num_classes)
        # connected = torch.tanh(connected)
        #
        # res = torch.sum(self.W * connected, 2)
        # res = res + self.B
        ###################################################################################################

        return res, group_loss

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        for i in range(self.num_classes):
            self.W[i].data.uniform_(-stdv, stdv)
        if self.B is not None:
            for i in range(self.num_classes):
                self.B[i].data.uniform_(-stdv, stdv)

    def get_config_optim(self, lr):
        return [
            {'params': self.resnet_101.parameters(), 'lr': lr * 0.1},
            {'params': self.label_embed.parameters(), 'lr': lr},
            {'params': self.group_embed.parameters(), 'lr': lr},
            {'params': self.ProjectM.parameters(), 'lr': lr},
            {'params': self.fc_label.parameters(), 'lr': lr},
            {'params': self.fc1.parameters(), 'lr': lr},
            {'params': self.fc2.parameters(), 'lr': lr},
            {'params': self.W, 'lr': lr},
            {'params': self.B, 'lr': lr},
        ]
