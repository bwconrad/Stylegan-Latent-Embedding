''' Modified from: https://github.com/adamian98/pulse '''

import torch

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, l2_weight=1, l1_weight=0, geocross_weight=0):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]

        self.ref_im = ref_im
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight 
        self.geocross_weight = geocross_weight


    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return ((gen_im - ref_im).pow(2).mean((1, 2, 3)).clamp(min=0).sum())

    def _loss_l1(self, gen_im, ref_im, **kwargs):
        return ((gen_im - ref_im).abs().mean((1, 2, 3)).clamp(min=0).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    def forward(self, latent, gen_im):
        var_dict = {'latent': latent,
                    'gen_im': gen_im,
                    'ref_im': self.ref_im,
                    }

        loss = 0
        losses = {}
        if self.l2_weight>0:
            tmp_loss = self._loss_l2(**var_dict)
            losses['L2'] = tmp_loss
            loss += float(self.l2_weight)*tmp_loss
        if self.l1_weight>0:
            tmp_loss = self._loss_l1(**var_dict)
            losses['L1'] = tmp_loss
            loss += float(l1_weight)*tmp_loss
        if self.geocross_weight>0:
            tmp_loss = self._loss_geocross(**var_dict)
            losses['Geocross'] = tmp_loss
            loss += float(geocross_weight)*tmp_loss
        
        return loss, losses
