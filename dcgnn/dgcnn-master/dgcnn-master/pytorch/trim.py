from MetaPlugins import query_points_in_box, OctreePartition
import numpy as np
from CloudGenerator import point_cloud, params
from test_file import my_knn
import torch
point_cloud = point_cloud()
x = point_cloud
test = OctreePartition(x,128)
res = my_knn(test,2)
def trim(res):
    filtered_batch = []
    for i in range(res[0].shape[0]):
        batch_filtered = []
        for k in res:
            batch_filtered.append(k[i][k[i][:, 0] != -1])
        result = torch.cat(batch_filtered, dim=0)
        filtered_batch.append(result)
    return filtered_batch
