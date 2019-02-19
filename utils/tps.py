import torch
import torch.nn.functional as F

def tps_grid(H, W):
    xi = torch.linspace(-1, 1, H)
    yi = torch.linspace(-1, 1, W)

    xx, yy = torch.meshgrid(xi, yi)
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    return grid


def tps_dist_mat(grid1, grid2):
    D = grid1.reshape(-1, 1, 2) - grid2.reshape(1, -1, 2)
    D = torch.sum(D ** 2., 2)
    return D


def tps(H, W, NKp):
    # BxNc+3x2x1
    npixels = H * W
    grid_pixels = tps_grid(100, 100)
    grid_kp = tps_grid(10, 10)
    D = tps_dist_mat(grid_pixels, grid_kp)
    D = D * torch.log(D + 1e-5)

    L = torch.cat((D, torch.ones(npixels, 1), grid_pixels), 1)

    aff = torch.tensor([[0.,0],[1,0],[0,1]])
    A = torch.cat((torch.randn(NKp*NKp,2)*0.01, aff),0)

    grid = torch.matmul(L, A)

    return grid.reshape(1,H,W,2)




if __name__ == '__main__':

    import pylab

    im = torch.randn(1,3,100,100) + 0.5
    grid = tps(100,100,10)
    imw = F.grid_sample(im, grid)

    pylab.imshow(imw.permute(0,2,3,1)[0])
    pylab.show()
