import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

dtype = torch.cuda.FloatTensor


class estimate_alpha(nn.Module):
    def __init__(self, init_value = 0.2):
        super(estimate_alpha, self).__init__()
        
        self.alpha = torch.nn.Parameter(torch.tensor(init_value, device = 'cuda:0', requires_grad = True)).type(dtype)
        
    def forward(self):
        return self.alpha
    
    
class optimal_kernel(nn.Module):
    def __init__(self, max_val = 0.2, init_value = None, order_up_to = 5, piston_tip_tilt = False):
        super(optimal_kernel, self).__init__()
        
        if init_value == None:
            if not piston_tip_tilt:
                _k4 = torch.rand((1, ), device = 'cuda:0', requires_grad = True)
                
            else:
                _k4 = torch.rand((4, ), device = 'cuda:0', requires_grad = True)
                
            _k5 = torch.full((1, ), fill_value = 0.5, device = 'cuda:0', requires_grad = True)
            
            if order_up_to == 4:
                _k6_15 = torch.rand((10, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_15), 0) - max_val).type(dtype)
                
            elif order_up_to == 5:
                _k6_21 = torch.rand((16, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_21), 0) - max_val).type(dtype)
                
            elif order_up_to == 6:
                _k6_28 = torch.rand((23, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_28), 0) - max_val).type(dtype)
                
            elif order_up_to == 7:
                _k6_36 = torch.rand((31, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_36), 0) - max_val).type(dtype)
                
            elif order_up_to == 8:
                _k6_45 = torch.rand((40, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_45), 0) - max_val).type(dtype)
                
            elif order_up_to == 9:
                _k6_55 = torch.rand((50, ), device = 'cuda:0', requires_grad = True)

                self.k = torch.nn.Parameter(2 * max_val * torch.cat((_k4, _k5, _k6_55), 0) - max_val).type(dtype)
            
        else:
            self.k = torch.nn.Parameter(torch.tensor(init_value, device = 'cuda:0', requires_grad = False)).type(dtype)
        
    def forward(self):
        return self.k


class LinearNet(nn.Module):
    def __init__(self, input_ch, W, D_in):
        super(LinearNet, self).__init__()
        
        self.input_ch = input_ch
        self.W = W
        self.D_in = D_in
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D_in-1)]
        )
        
        self.last_layer = nn.Linear(W, input_ch)
        
    def forward(self, x):
        h = x
        
        for i, _ in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h)
        
        outputs = self.last_layer(h)
        
        return outputs
    
    
def weights_init_kernel(m):
    classname = m.__class__.__name__  
    
    if classname.find('Linear') != -1:
        # nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        nn.init.xavier_normal_(m.weight.data, gain=0.1)
        # nn.init.constant_(m.weight.data, 1/12)
        nn.init.zeros_(m.bias.data)


def input_coord_2d(image_width, image_height):
    tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = rearrange(mgrid, 'w h c -> (w h) c')
    
    return mgrid


def input_coord_3d(image_depth, image_width, image_height):
    tensors = [torch.linspace(-1, 1, steps = image_depth), torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = rearrange(mgrid, 'd w h c -> (d w h) c')
    
    return mgrid


def radial_encoding(in_node, dia_digree, L_xy):
    s = torch.sin(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    c = torch.cos(torch.arange(0, 180, dia_digree) * np.pi / 180).unsqueeze(-1).cuda(0)
    
    fourier_mapping = torch.cat((s, c), axis = -1).T
    xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
    
    # Accommodate float values for L_xy.
    for l in range(int(np.floor(L_xy))):
        cur_freq = torch.cat((torch.sin(2 ** (l+L_xy-np.floor(L_xy)) * np.pi * xy_freq), torch.cos(2 ** (l+L_xy-np.floor(L_xy)) * np.pi * xy_freq)), axis = -1)
        
        if l == 0:
            tot_freq = cur_freq
        else:
            tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
    
    # for l in range(L_z):
    #     cur_freq = torch.cat((torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)), torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))), axis = -1)
    #     tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
        
    
    return tot_freq


def triangular_radial_encoding(in_node, dia_digree, L_xy):
    s = (2 * torch.abs(torch.remainder(torch.arange(0, 180, dia_digree) / 180 - 0.5, 2) - 1) - 1).unsqueeze(-1).cuda(0)
    c = (2 * torch.abs(torch.remainder(torch.arange(0, 180, dia_digree) / 180, 2) - 1) - 1).unsqueeze(-1).cuda(0)
    
    fourier_mapping = torch.cat((s, c), axis = -1).T
    xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)
    
    for l in range(L_xy):
        # cur_freq = torch.cat((torch.sin(2 ** l * np.pi * xy_freq), torch.cos(2 ** l * np.pi * xy_freq)), axis = -1)
        cur_freq = torch.cat(((2 * torch.abs(torch.remainder(2 ** l * xy_freq - 0.5, 2) - 1) - 1), 
                              (2 * torch.abs(torch.remainder(2 ** l * xy_freq, 2) - 1) - 1)), axis = -1)

        if l == 0:
            tot_freq = cur_freq
        else:
            tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
    
    # for l in range(L_z):
    #     cur_freq = torch.cat((torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)), torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1))), axis = -1)
    #     tot_freq = torch.cat((tot_freq, cur_freq), axis = -1)
        
    
    return tot_freq
                    
                    
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
    
    
class TriangularWaveEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(TriangularWaveEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.out_channels = in_channels*(2*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            out += [2 * torch.abs(torch.remainder(freq*x - 0.5, 2) - 1) - 1]
            out += [2 * torch.abs(torch.remainder(freq*x, 2) - 1) - 1]
            
        return torch.cat(out, -1)

    
# class NeRF(nn.Module):
#     def __init__(self,
#                  D=8, 
#                  W=256,
#                  skips=[4],
#                  in_channels = 63,
#                  out_channels = 180
#                  ):
#         """
#         D: number of layers for density (sigma) encoder
#         W: number of hidden units in each layer
#         skips: add skip connection in the Dth layer
#         """
#         super(NeRF, self).__init__()
#         self.D = D
#         self.W = W
#         self.skips = skips
#         # self.beta = beta
#         # self.max_val = max_val

#         self.pre = nn.Sequential(
#             nn.Linear(in_channels, W),
#             nn.ReLU(True))
        
#         # xyz encoding layers
#         for i in range(D):
#             if i == 0:
#                 layer = nn.Linear(W, W)
#             elif i in skips:
#                 layer = nn.Linear(W+W, W)
#             else:
#                 layer = nn.Linear(W, W)
#             layer = nn.Sequential(layer, nn.ReLU(True))
#             setattr(self, f"xyz_encoding_{i+1}", layer)

#         # output layers
#         self.post = nn.Sequential(
#             nn.Linear(W+W, W),
#             nn.ReLU(True),
#             nn.Linear(W, out_channels))
        
#     def forward(self, x):
#         """
#         Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
#         For rendering this ray, please see rendering.py
#         Inputs:
#             x: (B, self.in_channels_xyz(+self.in_channels_dir))
#                the embedded vector of position and direction
#             sigma_only: whether to infer sigma only. If True,
#                         x is of shape (B, self.in_channels_xyz)
#         Outputs:
#             if sigma_ony:
#                 sigma: (B, 1) sigma
#             else:
#                 out: (B, 4), rgb and sigma
#         """
#         input_xyz = x        
#         xyz_pre = self.pre(input_xyz)
#         xyz_ = xyz_pre
        
#         for i in range(self.D):
#             if i in self.skips:
#                 xyz_ = torch.cat([xyz_pre, xyz_], -1)
                
#             xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

#         xyz_ = torch.cat([xyz_pre, xyz_], -1)
#         obj = self.post(xyz_)
        
#         # obj = nn.Softplus(beta = self.beta)(obj) if self.beta is not None else self.max_val * nn.Sigmoid()(obj)

#         return obj


class NeRF(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 skips=[4],
                 in_channels = 63,
                 out_channels = 180
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        # self.beta = beta
        # self.max_val = max_val

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.enc_last = nn.Linear(W, W)

        # output layers
        self.post = nn.Sequential(
                        nn.Linear(W, W//2),
                        nn.ReLU(True),
                        nn.Linear(W//2, out_channels))
        
    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        enc_last = self.enc_last(xyz_)
        obj = self.post(enc_last)
        
        # obj = nn.Softplus(beta = self.beta)(obj) if self.beta is not None else self.max_val * nn.Sigmoid()(obj)

        return obj
    
    
class NeRF_Dropout(nn.Module):
    def __init__(self,
                 D=8, 
                 W=256,
                 skips=[4],
                 in_channels = 63,
                 out_channels = 180,
                 dropout_rate = 0.5
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        """
        super(NeRF_Dropout, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        # self.beta = beta
        # self.max_val = max_val
        
        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels, W)
            else:
                layer = nn.Linear(W, W)
            
            if i%2 == 0 and i > 0:
                layer = nn.Sequential(nn.Dropout(dropout_rate), layer, nn.ReLU(True))
            else:
                layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
            
        self.enc_last = nn.Linear(W, W)

        # output layers
        self.post = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(W, W//2),
                        nn.ReLU(True),
                        nn.Linear(W//2, out_channels))
                
    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        enc_last = self.enc_last(xyz_)
        obj = self.post(enc_last)
        
        # obj = nn.Softplus(beta = self.beta)(obj) if self.beta is not None else self.max_val * nn.Sigmoid()(obj)

        return obj
