#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k, chunk_size=1024):
    batch_size, _, num_points = x.size()
    idx_list = []

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)

        # Select a chunk of points
        chunk = x[:, :, start:end]  # (B, 3, chunk_size)

        # Compute pairwise distances within the chunk
        dist = -2 * torch.matmul(chunk.transpose(2, 1), x)  # (B, chunk_size, N)
        dist += torch.sum(chunk ** 2, dim=1, keepdim=True).transpose(2, 1)  # (B, chunk_size, N)
        dist += torch.sum(x ** 2, dim=1, keepdim=True)  # (B, chunk_size, N)

        # Get top-k indices for the chunk
        idx = dist.topk(k=k, dim=-1, largest=False)[1]  # (B, chunk_size, k)
        idx_list.append(idx)

    # Concatenate indices for all chunks
    idx = torch.cat(idx_list, dim=1)  # (B, N, k)
    return idx


def get_graph_feature_delayed_chunked(x, features, k=20, chunk_size=1024):
    """
    Delayed Aggregation: Separates feature computation and aggregation.
    """
    batch_size, num_dims, num_points = x.size()

    # Perform chunked KNN
    idx = knn(x, k=k, chunk_size=chunk_size)  # (B, N, k)

    # Flatten batch and point indices
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # Gather features for neighbor points
    feature_neighbors = features.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[idx, :]
    feature_neighbors = feature_neighbors.view(batch_size, num_points, k, -1)

    # Combine features with central point features
    central_features = features.transpose(2, 1).unsqueeze(2).expand_as(feature_neighbors)
    combined_features = torch.cat((feature_neighbors - central_features, central_features), dim=-1)

    return combined_features.permute(0, 3, 1, 2).contiguous()  # (B, 2*num_dims, N, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x



def prune_weights(layer, threshold):
    """
    Prune weights of a layer below a certain threshold.
    """
    with torch.no_grad():
        mask = torch.abs(layer.weight) > threshold
        layer.weight.mul_(mask)


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        # Layer 1
        features = x
        x = get_graph_feature_delayed_chunked(x, features, k=self.k, chunk_size=1024)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # Layer 2
        x = get_graph_feature_delayed_chunked(x1, x1, k=self.k, chunk_size=1024)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # Layer 3
        x = get_graph_feature_delayed_chunked(x2, x2, k=self.k, chunk_size=1024)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # Layer 4
        x = get_graph_feature_delayed_chunked(x3, x3, k=self.k, chunk_size=1024)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # Concatenate features and final layers
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Global pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # Fully connected layers
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

    def prune_model(self, threshold):
        """
        Apply pruning to all convolutional and linear layers in the model.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                prune_weights(module, threshold)


def get_statistics(self):
        return np.array(self.stats)