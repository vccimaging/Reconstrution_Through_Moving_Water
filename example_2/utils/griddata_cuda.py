import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

from .griddata_cuda_python import Griddata_Cuda_Python


class Griddata_Cuda_Function(Function):

    @staticmethod
    def forward(ctx, im0, grid):

        ctx.save_for_backward(im0, grid)
        if im0.is_cuda:
            im1 = Griddata_Cuda_Python.forward(im0, grid)
        else:
            im1 = Griddata_Cuda_Python.forward(im0, grid)

        return im1

    @staticmethod
    def backward(ctx, grad_output):
        im0, grid = ctx.saved_variables
        if grad_output.is_cuda:
            im0_grad, grid_grad = Griddata_Cuda_Python.backward(
                grad_output, im0, grid)
        else:
            im0_grad, grid_grad = Griddata_Cuda_Python.backward(
                grad_output, im0, grid)
        return im0_grad, grid_grad, None


class Griddata_Cuda(Module):

    def __init__(self):
        super(Griddata_Cuda, self).__init__()

    def forward(self, im0, grid):
        return Griddata_Cuda_Function.apply(im0, grid)
