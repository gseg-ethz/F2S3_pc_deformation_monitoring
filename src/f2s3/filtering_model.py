from src.f2s3.utils import kabsch_transformation_estimation
import torch
import torch.nn as nn

class PointCN(nn.Module):
    def __init__(self, channels):
        super(PointCN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels, eps=1e-3,affine=False, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels, eps=1e-3,affine=False, track_running_stats=False),
                nn.ReLU()
                )

    def forward(self, x):

        out = self.conv(x)

        return out + x

class FilteringNetwork(nn.Module): 


    def __init__(self):
        super(FilteringNetwork, self).__init__()

        numlayer = 12
        nchannel = 128

        self.l1 = nn.Conv2d(6, nchannel, kernel_size=1)
        self.l2 = []
        for _ in range(numlayer):
            self.l2.append(PointCN(nchannel))


        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(nchannel, 1, kernel_size=1)

        self.activation = nn.ReLU(inplace=True)

    def compute_weights(self, x):

        # First pass through the network
        assert x.dim() == 4 and x.shape[1] == 1
        #data: b*1*n*c
        x = x.transpose(1, 3)


        x1 = self.l1(x)
        x2 = self.l2(x1)
        out = self.output(x2).squeeze(-1).squeeze(1)

        return self.activation(torch.tanh(out))


    def filter_input(self, data, data_raw):

        robust_estimate = False
        # First pass through the network
        infered_weights = self.compute_weights(data)

        x1, x2 = data_raw[:, :, :3], data_raw[:, :, 3:]

        rotation_est, translation_est, residuals, _ = kabsch_transformation_estimation(
            x1, x2, infered_weights)

        inliers = torch.where(residuals < torch.median(residuals))[1]

        if inliers.shape[0] >= 5 and torch.median(residuals) < 0.5:
            robust_estimate = True
            weights = torch.zeros_like(infered_weights)
            weights[0,inliers.reshape(-1)] = 1.0

            rotation_est, translation_est, residuals, _ = kabsch_transformation_estimation(
            x1, x2, weights)

        output = {}
        output['scores'] = infered_weights
        output['rot_est'] = rotation_est.squeeze(0)
        output['trans_est'] = translation_est.squeeze(0)
        output['robust_estimate'] = robust_estimate

        return output






if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    test = FilteringNetwork()

    print(count_parameters(test))