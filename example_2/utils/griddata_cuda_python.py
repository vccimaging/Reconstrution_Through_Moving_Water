import torch
from torch.nn import Module, Parameter
from torch.autograd import Function


class Griddata_Cuda_Python:
    @staticmethod
    def forward(im0, grid):
        im1 = torch.zeros_like(im0)
        H = im0.shape[0]
        W = im0.shape[1]
        B = im0.shape[2]
        round_grid = torch.round(grid).type(torch.IntTensor)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = round_grid[h,w,b,0]
                    y = round_grid[h,w,b,1]
                    if x >= 0 and x < W and y >= 0 and y < H:
                        im1[y, x, b, :] = im0[h, w, b, :]
                        
        return im1

    @staticmethod
    def backward(grad_output, im0, grid):
        
        H = im0.shape[0]
        W = im0.shape[1]
        B = im0.shape[2]
        
        im0_grad = torch.zeros_like(grad_output)
        grid_grad = None
        round_grid = torch.round(grid).type(torch.IntTensor)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = round_grid[h,w,b,0]
                    y = round_grid[h,w,b,1]
                    if x >= 0 and x < W and y >= 0 and y < H:
                        im0_grad[h, w, b, :] = grad_output[y, x, b, :]
                        
        return im0_grad, grid_grad
