import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np
import tensorflow as tf
def query_points_in_box(coords, point_cloud, box_min, box_max):
    """
    查询该点是否落到指定区域
    Args:
        point_cloud: 点云全集，original x
        box_min: 指定区域最小边界地张量，定义box的最小坐标 （num_dim,）
        box_max: 指定区域最大边界张量，定义box的最大坐标  （num_dim,）

    Returns:
        一个与点云维度相同的张量，是落在指定区域的全部点坐标

    """
    box_min = box_min.unsqueeze(2)
    box_max = box_max.unsqueeze(2)
    # Create masks for points within the bounding box
    mask = (coords>=box_min)&(coords<=box_max)
    # Check if all dimensions of a point are within the box
    mask = mask.all(dim=1)
    # Filtering
    filtered_point_cloud = []
    # lengths = []
    for i in range(point_cloud.shape[0]):
        filtered_batch = point_cloud[i][:,mask[i]].transpose(0,1)
        filtered_point_cloud.append(filtered_batch)
        # lengths.append(len(filtered_batch))
    # Stack the results back into a batch
    # length = [len(s) for s in filtered_point_cloud]
    result = pad_sequence(filtered_point_cloud, batch_first=True, padding_value=0).transpose(1,2)
    # result = torch.stack(filtered_point_cloud)
    return result
    # length = [len(s) for s in filtered_point_cloud]
    # max_len = max(length)
    # tensor1_padded = F.pad(filtered_point_cloud[0],(0,max_len-filtered_point_cloud[0].shape[0]),"constant",0)
    # tensor2_padded = F.pad(filtered_point_cloud[1],(0,max_len-filtered_point_cloud[1].shape[0]),"constant",0)
    # result = torch.stack([tensor1_padded,tensor2_padded],dim=0)
def OctreePartition(point_cloud, max_points_per_block):
    """
    Partitions a point cloud into smaller blocks using an octree_based partitioning method
    to manage and limit the number of points in each block

    Args:
        point_cloud (torch.Tensor): A 3D tensor representing the point cloud data with shape
        (batch_size, 3, num_points). Each point has x,y and z coordinates.
        max_points_per_block: The maximum allowed number of points in each partitioned block.
        If the number of poitns in the current block exceeds this limit, the block will be further
        subdivided.

    Returns:
        list[torch.Tensor]: A list of smaller point cloud blocks, each represented as a 3D tensor
        with the same format as 'point_cloud'. The subdivision process continues until all
        blocks contain no more than 'max_points_per_block' points.

    """
    coords = point_cloud.clone()
    num_dims = point_cloud.shape[1]
    num_pieces = num_dims//3
    block1 = point_cloud[:,0:num_pieces,:]
    block2 = point_cloud[:,num_pieces:2*num_pieces,:]
    block3 = point_cloud[:,2*num_pieces:,:]
    feat1 = (block1 ** 2).sum(dim=1, keepdim=True)
    feat2 = (block2 ** 2).sum(dim=1, keepdim=True)
    feat3 = (block3 ** 2).sum(dim=1, keepdim=True)
    coords = torch.cat([feat1,feat2,feat3],axis=1)
    num_points_in_block = point_cloud.shape[2]
    if num_points_in_block <= max_points_per_block:
        return [point_cloud]
    else:
        tensor_for_min = coords.clone()
        tensor_for_min[tensor_for_min==0] = float('inf')
        tensor_for_max = coords.clone()
        tensor_for_max[tensor_for_max==0] = float('-inf')
        bbox_min = torch.min(tensor_for_min, dim=2).values
        bbox_max = torch.max(tensor_for_max, dim=2).values
        bbox_mid = (bbox_min + bbox_max) / 2
        a = bbox_max
        i = bbox_min
        d = bbox_mid
        query_min = d
        query_max = a
        query_points1 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block1 = OctreePartition(query_points1, max_points_per_block)

        # # print(query_points1.shape[2] >= k)
        # if query_points1.shape[2] >= k:
        #     k1 = my_knn(query_points1, k)
        # else:
        #     k1 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)
        # # print(my_knn(query_points1, k))

        query_min = torch.stack([d[:, 0], d[:, 1], i[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([a[:, 0], a[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_points2 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block2 = OctreePartition(query_points2, max_points_per_block)

        # if query_points2.shape[2] >= k:
        #     k2 = my_knn(query_points2, k)
        # else:
        #     k2 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)
        # # print(my_knn(query_points2, k))

        query_min = torch.stack([d[:, 0], i[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([a[:, 0], d[:, 1], a[:, 2]], dim=1).to(point_cloud.device)
        query_points3 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block3 = OctreePartition(query_points3, max_points_per_block)
        # if query_points3.shape[2] >= k:
        #     k3 = my_knn(query_points3, k)
        # else:
        #     k3 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)

        query_min = torch.stack([d[:, 0], i[:, 1], i[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([a[:, 0], d[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_points4 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block4 = OctreePartition(query_points4, max_points_per_block)
        # if query_points4.shape[2] >= k:
        #     k4 = my_knn(query_points4, k)
        # else:
        #     k4 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)

        query_min = torch.stack([i[:, 0], d[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([d[:, 0], a[:, 1], a[:, 2]], dim=1).to(point_cloud.device)
        query_points5 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block5 = OctreePartition(query_points5, max_points_per_block)
        # if query_points5.shape[2] >= k:
        #     k5 = my_knn(query_points5, k)
        # else:
        #     k5 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)

        query_min = torch.stack([i[:, 0], d[:, 1], i[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([d[:, 0], a[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_points6 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block6 = OctreePartition(query_points6, max_points_per_block)
        # if query_points6.shape[2] >= k:
        #     k6 = my_knn(query_points6, k)
        # else:
        #     k6 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)

        query_min = torch.stack([i[:, 0], i[:, 1], d[:, 2]], dim=1).to(point_cloud.device)
        query_max = torch.stack([d[:, 0], d[:, 1], a[:, 2]], dim=1).to(point_cloud.device)
        query_points7 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block7 = OctreePartition(query_points7, max_points_per_block)
        # if query_points7.shape[2] >= k:
        #     k7 = my_knn(query_points7, k)
        # else:
        #     k7 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)

        query_min = i
        query_max = d
        query_points8 = query_points_in_box(coords, point_cloud, query_min, query_max)
        block8 = OctreePartition(query_points8, max_points_per_block)
        # if query_points8.shape[2] >= k:
        #     k8 = my_knn(query_points8, k)
        # else:
        #     k8 = torch.empty(point_cloud.shape[0], 3, 0, device=point_cloud.device)
        # return torch.cat((k1,k2,k3,k4,k5,k6,k7,k8),dim=1)
        return block1+block2+block3+block4+block5+block6+block7+block8
def my_knn(partitions, k):
    """
    Computes the k-nearest neighbors for each point in the given partitions of a point cloud.

    Args:
        partitions (list[torch.Tenosr): A list of 3D tensors, where each tensor represents a
        partitioned block of the point cloud with shape (batch_size, 3, num_points). Each point
        has x, y and z coordinates. The partitions are used to calculate pairwise distances
        and determine the k-nearest neightbors.
        k (int): The number of nearest neighbors to compute for each point.

    Returns:
        list[torch.Tensor]: A list of tensors, where each tensor contains the indices of the k-nearest
        neighbors for each point in the corresponding partition. The shape of each tensor is
        (batch_size, num_points, k). Indices corresponding to masked (invalid) points are set to -1.

    """
    batch_size= partitions[0].shape[0]
    labels = np.zeros(batch_size)
    idxs = []
    for x in partitions:
        # for label modifications
        expanded_labels = [torch.full((x.shape[2], k), value) for value in labels]
        const = torch.stack(expanded_labels)
        # for KNN calculations
        mask = torch.all(x == 0, dim=1, keepdim=True)
        x_masked = torch.where(mask, torch.ones_like(x) * float('inf'), x)
        pairwise_distance = -torch.cdist(x_masked.transpose(1,2),x_masked.transpose(1,2))
        pairwise_distance = torch.nan_to_num(pairwise_distance, nan=-1.0, posinf=-1.0, neginf=-1.0)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]+const
        invalid_idx = torch.full((x.size(0), x.size(2), k), -1, dtype=torch.long, device=x.device)
        idx = torch.where(mask.squeeze(1).unsqueeze(-1).expand_as(idx), invalid_idx, idx)
        idxs.append(idx)
        labels += (~mask).sum(dim=-1).squeeze(1).tolist()
    return idxs