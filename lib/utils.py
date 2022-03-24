import torch
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters
    
    Args:
        x1  (np array): points of the point cloud [n,3]
        R   (np array): estimated rotation matrice [3,3]
        t   (np array): estimated translation vectors [3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [n,3]
    """

    if torch.is_tensor(x1):
        x1_t = (torch.matmul(R, x1.permute(1,0)) + t).permute(1,0)
    else:
        x1_t = (np.matmul(R, x1.transpose()) + t).transpose()

    return x1_t

def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res
    
def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimate rotation matrix is then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.

    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)

    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    weight_matrix = torch.diag_embed(weights.squeeze(2))

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           torch.matmul(weight_matrix, x2_centered))

    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


def compute_c2c(source_pc, target_pc):
    """
    Cloud to cloud distance computation.

    Args:
        source_pc            (np array): points of the source point cloud [n,3]
        target_pc            (np array): points of the target point cloud [m,3]

    Returns:
        c2c_dist  (np array): cloud to cloud distances [n]
    """

    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    neigh.fit(target_pc)

    c2c_dist, _ = neigh.kneighbors(source_pc, n_neighbors=1, return_distance=True)

    return c2c_dist