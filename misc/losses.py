import torch

dtype = torch.cuda.FloatTensor

def tv_1d(img, axis = 'z', normalize = False):
    if axis == 'z':
        if not normalize:
            _variance = torch.sum(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
        else:
            _variance = torch.mean(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
            
    elif axis == 'y':
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
        else:
            _variance = torch.mean(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
            
    else:
        if not normalize:
            _variance = torch.sum(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
        else:
            _variance = torch.mean(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
            
    return _variance
   
    
def tv_2d(img):
    # img: D x W x H
    h_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 1)))
    w_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 2)))
    
    return h_variance + w_variance


def tv_3d(img, weights):
    # d_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 0)))
    # h_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 1)))
    # w_variance = torch.sum(torch.abs(img - torch.roll(img, 1, dims = 2)))
    d_variance = torch.sum(torch.abs(img[0:-1, :, :] - img[1::, :, :]))
    h_variance = torch.sum(torch.abs(img[:, 0:-1, :] - img[:, 1::, :]))
    w_variance = torch.sum(torch.abs(img[:, :, 0:-1] - img[:, :, 1::]))
    
    return weights[0] * d_variance + weights[1] * h_variance + weights[2] * w_variance


def DnCNN_loss(model, inp, tar, rnd):
    with torch.no_grad():
        return ((model(inp[rnd, :, :].unsqueeze(1) * 2**8) - model(tar[rnd, :, :].unsqueeze(1) * 2**8))**2).mean()
    

def npcc_loss(y_pred, y_true):
    up = torch.mean((y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true)))
    down = torch.std(y_pred) * torch.std(y_true)
    loss = 1.0 - up / down

    return loss.type(dtype)


def ssim_loss(img1, img2):
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_mu2 = mu1 * mu2
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    
    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    loss = 1.0 - (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / torch.clamp((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min = 1e-8, max = 1e8))
        
    return loss.type(dtype)


def nonlinear_diffusion_loss(img, upper_bound):
    h_abs_grad = torch.abs(img[:, :, :-1] - img[:, :, 1:])
    w_abs_grad = torch.abs(img[:, :-1, :] - img[:, 1:, :])
    
    def reg_fun(abs_grad, ubd):
        return torch.minimum(abs_grad, torch.full_like(abs_grad, ubd))
        
    loss_h = torch.sum(reg_fun(h_abs_grad, upper_bound))
    loss_w = torch.sum(reg_fun(w_abs_grad, upper_bound))
    
    return (loss_h + loss_w)


def tv_range_loss(img, lower_bound):
    h_abs_grad = torch.abs(img[:, :, :-1] - img[:, :, 1:])
    w_abs_grad = torch.abs(img[:, :-1, :] - img[:, 1:, :])
    
    def reg_fun(abs_grad, lbd):
        return torch.maximum(abs_grad, torch.full_like(abs_grad, lbd))
        
    loss_h = torch.sum(reg_fun(h_abs_grad, lower_bound))
    loss_w = torch.sum(reg_fun(w_abs_grad, lower_bound))
    
    return (loss_h + loss_w)


def second_order_diff_loss(img, upper_bound):
    second_order_diff = torch.abs(img[:, :-1, :-1] - img[:, :-1, 1:] - img[:, 1:, :-1] + img[:, 1:, 1:])
    
    def reg_fun(diff, lbd):
        return torch.minimum(diff, torch.full_like(diff, lbd))
        
    loss = torch.sum(reg_fun(second_order_diff, upper_bound))
    
    return loss


def relative_l1_loss(y_pred, y_true, val):
    return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_pred.detach()) + val))


def relative_l2_loss(y_pred, y_true, val):
    return torch.mean((y_pred - y_true)**2 / (y_pred.detach()**2 + val))


def fourier_loss(F1, F2):
    projection = torch.abs(F1 * torch.conj(F2)) / torch.abs(F1) / torch.abs(F2)
    
    return 1.0 - projection.mean()


# def FRCloss(img1, img2):
#     nz, nx, ny = [torch.tensor(i) for i in img1.shape]
#     rnyquist = nx//2
    
#     xx = torch.cat((torch.arange(0, nx/2), torch.arange(-nx/2, 0)))
#     yy = xx
#     X, Y = torch.meshgrid(xx, yy)
#     R = X ** 2 + Y ** 2
#     index = torch.round(torch.sqrt(R.float()))
#     r = torch.arange(0, rnyquist + 1)
    
#     F1 = fftn(img1).permute(1, 2, 0)
#     F2 = fftn(img2).permute(1, 2, 0)
    
#     C_r, C1, C2, C_i = [torch.empty(rnyquist + 1, nz) for i in range(4)]

#     for ii in r:
#         auxF1 = F1[torch.where(index == ii)]; aF1r = torch.real(auxF1); aF1i = torch.imag(auxF1)
#         auxF2 = F2[torch.where(index == ii)]; aF2r = torch.real(auxF2); aF2i = torch.imag(auxF2)

#         C_r[ii] = torch.sum(aF1r * aF2r + aF1i * aF2i, 0)
#         C_i[ii] = torch.sum(aF1i * aF2r - aF1r * aF2i, 0)
#         C1[ii] = torch.sum(aF1r ** 2 + aF1i ** 2, 0)
#         C2[ii] = torch.sum(aF2r ** 2 + aF2i ** 2, 0)

#     FRC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
#     FRCm = 1.0 - torch.where(FRC != FRC, torch.tensor(1.0), FRC)
#     FRCloss = torch.mean((FRCm) ** 2)    
