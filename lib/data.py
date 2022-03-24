import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d

_EPS = 1e-6

class FeatureExtractionDataset(data.Dataset):
    def __init__(self, data, data_overlap, points_per_batch, feature_radius, num_points=256):
        
        self.data = data
        self.data_overlap = data_overlap
        self.points_per_batch = points_per_batch
        self.feature_radius = feature_radius
        self.num_points = num_points

        self.pcd_tree = o3d.geometry.KDTreeFlann(self.data_overlap)
        self.cnt = 0 

    def __getitem__(self, idx):

        patches_batch = []

        offset = idx * self.points_per_batch

        batch_data = np.asarray(self.data.points)[offset:offset + self.points_per_batch,:]

        for pt in batch_data:
            pts = self.extract_patch(pt)
            patches_batch.append(torch.from_numpy(pts.T).unsqueeze(0))


        return torch.cat(patches_batch, axis=0).float()

    def extract_patch(self, pt):

        _, patch_idx, patch_dist = self.pcd_tree.search_radius_vector_3d(pt, radius=self.feature_radius)

        ptnn = np.asarray(self.data_overlap.points)[patch_idx[1:], :].T
        patch_dist = np.sqrt(np.array(patch_dist)[1:])
        ptall = np.asarray(self.data_overlap.points)[patch_idx, :].T

        if ptall.shape[1] > 10:

            vect_diff = ptnn - pt[:, np.newaxis]

            # eq. 3
            ptnn_cov = 1 / ptnn.shape[0] * np.dot(vect_diff, vect_diff.T)

            # The normalized (unit “length”) eigenvectors, s.t. the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
            a, v = np.linalg.eig(ptnn_cov)
            smallest_eigevalue_idx = np.argmin(a)
            np_hat = v[:, smallest_eigevalue_idx]

            # eq. 4
            zp = np_hat if np.sum(np.dot(np_hat, -vect_diff)) > 0 else - np_hat

            v = vect_diff - (np.dot(vect_diff.T, zp[:, np.newaxis]) * zp).T
            alpha = (self.feature_radius - patch_dist)** 2
            beta = np.dot(vect_diff.T, zp[:, np.newaxis]).squeeze() ** 2

            # e.q. 5
            if np.abs(np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis]))) < _EPS: 
                xp = 1 / (np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis])) + _EPS) * np.dot(v, (alpha * beta)[:, np.newaxis])
            else:
                xp = 1 / np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis])) * np.dot(v, (alpha * beta)[:, np.newaxis])

            xp = xp.squeeze()

            yp = np.cross(xp, zp)

            lRg = np.asarray([xp, yp, zp]).T

            # rotate w.r.t local frame and centre in zero using the chosen point
            ptall = (lRg.T @ (ptall - pt[:, np.newaxis])).T

            # this is our normalisation
            ptall /= self.feature_radius

            T = np.zeros((4, 4))
            T[-1, -1] = 1
            T[:3, :3] = lRg
            T[:3, -1] = pt

        else:
            ptall = ptall.T

            # this is our normalisation
            ptall /= self.feature_radius

        # to make sure that there are at least self.patch_size points, pad with zeros if not
        if ptall.shape[0] < self.num_points:
            ptall = np.concatenate((ptall, np.zeros((self.num_points - ptall.shape[0], 3))))

        inds = np.random.choice(ptall.shape[0], self.num_points, replace=False)

        return ptall[inds]



    def __len__(self):
        return int(np.ceil(np.asarray(self.data.points).shape[0]/self.points_per_batch))

    def reset_seed(self,seed=41):
        logging.info('Resetting the data loader seed to {}'.format(seed))
        self.randng.seed(seed)