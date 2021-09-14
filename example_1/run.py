import torch
import torch.nn as nn
import cv2
import numpy as np

from math import ceil, floor
import time
from skimage import io
from scipy import sparse

from utils.griddata_cuda import Griddata_Cuda

class Param(object):
    def __init__(self, xPixel, yPixel, time_frame, ref):
        self.pixel_length = xPixel*yPixel
        #############################################################
        self.water2Camera = 20    # Distance from camera to flat water surface
        self.camera2Sensor = 1    # Distance from camera to sensor (normalize to 1)
        
        self.ylength = 10         # Length of the field-of-view in the Y-axis
        self.scale   = 2
        self.ref = ref
        #############################################################
        self.pixel_size = self.ylength / yPixel / self.water2Camera
        self.refra_index = 1.33
        self.time_frame = time_frame
        self.camera = np.array([ [0], [0], [0] ] )

def setImageCoor(xPixel, yPixel, uv, Param):
    
    pixel_size = Param.pixel_size
    x = np.arange(xPixel)
    y = np.arange(yPixel)
    xx, yy = np.meshgrid(x, y)
    xy = np.append( [xx.flatten()], [yy.flatten()], axis = 0 )
    
    if uv.shape[0]:
        uv = np.append( [uv[:,:,0].flatten()], [uv[:,:,1].flatten()], axis = 0 )
    else:
        uv = np.zeros( [2, xPixel*yPixel] )
        
    center = [ [(xPixel-1)/2], [(yPixel-1)/2] ]
    center_array = np.tile( center, (1, xPixel*yPixel) )
    
    image_coor = ( xy + uv - center_array )*pixel_size
    # the third dimension is 1 (vectical distance from nodal point to image plane)
    image_coor = np.append( image_coor, [np.ones(xPixel*yPixel)], axis = 0 )  
    return image_coor

class pointCloudProj(torch.nn.Module):
    def __init__(self, param):
        super(pointCloudProj, self).__init__()
        self.camera = torch.from_numpy( param.camera )
        self.refra_index = param.refra_index
        self.pixel_size = param.pixel_size
        self.pixel_length = param.pixel_length
        self.time_frame = param.time_frame
        self.scale = param.scale
        
    def forward(self, P, H, I, N, W, W_dx, W_dy, M):
        
        S = I*(torch.sparse.mm(W, H.view(-1,1)).view(1,-1).repeat(3,1))
        e = torch.div( camera.repeat(1, xPixel*yPixel*time_frame) - S, \
                torch.norm( camera.repeat(1, xPixel*yPixel*time_frame) - S, dim=0 ).view(1,-1).repeat(3,1) )

        dx = (torch.sparse.mm(W_dx, H.view(-1,1)) / self.pixel_size / self.scale).view(1,-1)
        dy = (torch.sparse.mm(W_dy, H.view(-1,1)) / self.pixel_size / self.scale).view(1,-1)
        N = torch.cat( (dx, dy, -I[0,:]*dx - I[1,:]*dy - \
                        torch.sparse.mm(W, H.view(-1,1)).view(1,-1)) , dim=0 )
        N = torch.div( N, torch.norm(N, dim=0).view(1,-1).repeat(3,1) )
        
        r = ( 1/refra_index * torch.sum(N*e,0) - torch.sqrt( 1 - (1/refra_index)**2 * \
                        ( 1-torch.sum(N*e,0)**2 ) ) ).view(1,-1).repeat(3,1)*N - 1/refra_index*e
        
        D = torch.sum( M.repeat(3,1) * ( P.repeat(1, time_frame) - S - \
            torch.sum( (P.repeat(1, time_frame)-S) * r, dim=0 ).repeat(3,1)*r )**2 )
    
        return D
    
################################################################################    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# reflection padding required
reflection_pad = nn.ReflectionPad2d(1)

# compute gradient of depth map
gradient_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
gradient_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)

kernel = torch.Tensor([ [0, 0, 0], [1, -1, 0], [0, 0, 0] ])
kernel = kernel.view((1,1,3,3))

gradient_x.weight.data = kernel
gradient_x.weight.requires_grad = False
gradient_x = gradient_x.float().to(device)

gradient_y.weight.data = torch.transpose(kernel, 2, 3)
gradient_y.weight.requires_grad = False
gradient_y = gradient_y.float().to(device)

################################################################################
# compute curvature of water surface
cur_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
cur_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)

kernel = torch.Tensor([ [0, 0, 0], [1, -2, 1], [0, 0, 0] ])
kernel = kernel.view((1,1,3,3))

cur_x.weight.data = kernel
cur_x.weight.requires_grad = False
cur_x = cur_x.float().to(device)

cur_y.weight.data = torch.transpose(kernel, 2, 3)
cur_y.weight.requires_grad = False
cur_y = cur_y.float().to(device)

################################################################################
folder = 'parameters'
ref = 20            # The 20-th frame serves as reference frame
time_frame = 100    # Total number of frames

img = io.imread('%s/ref.png'%(folder) )
H = img.shape[0]
W = img.shape[1]
# Load the precomputed optical flow (from each frame to the reference frame), and the confidence masks
flo = np.load('%s/uv_com_%d.npy'%(folder,time_frame) )
floMask = np.load('%s/flowMask_%d.npy'%(folder,time_frame) )

xPixel = W
yPixel = H
param = Param(xPixel, yPixel, time_frame, ref)

for i in range(time_frame):
    if i==0 and i==ref-1:
        image_cloud = torch.from_numpy( setImageCoor(xPixel, yPixel, np.array([]), param ) )
    elif i==0 and i!=ref-1:
        image_cloud = torch.from_numpy( setImageCoor(xPixel, yPixel, flo[:,:,:,i], param ) )
    elif i < ref-1:
        image_cloud = torch.cat( (image_cloud, torch.from_numpy( setImageCoor(xPixel, yPixel, flo[:,:,:,i] , param ) ) ), dim=1 )
    elif i == ref-1:
        image_cloud = torch.cat( (image_cloud, torch.from_numpy( setImageCoor(xPixel, yPixel, np.array([]) , param ) ) ), dim=1 )
    elif i > ref-1:
        image_cloud = torch.cat( (image_cloud, torch.from_numpy( setImageCoor(xPixel, yPixel, flo[:,:,:,i-1] , param ) ) ), dim=1 )
        
# point cloud initialization
height_map = torch.ones( (time_frame, xPixel*yPixel), requires_grad=False).float()*param.water2Camera
point_cloud = torch.zeros(3, xPixel*yPixel, requires_grad=False).float()
################# Initialize ################################
D = torch.ones(xPixel*yPixel).float()*param.water2Camera + 20
#############################################################
surface = image_cloud[:,(ref-1)*xPixel*yPixel:ref*xPixel*yPixel]*height_map[0,:].repeat(3,1)

refra_index = param.refra_index
camera = torch.from_numpy(param.camera).float()
camera = camera.repeat((1,xPixel*yPixel))
s1 = surface - camera
s1 = s1 / torch.norm(s1, dim = 0).view(1,-1).repeat(3,1)

N  = torch.from_numpy( np.array([0,0,1]) ).float()
N = N.view(-1,1).repeat(1,xPixel*yPixel)

s2 = ( 1/refra_index * torch.sum(-N*-s1,0) - torch.sqrt( 1 - (1/refra_index)**2 * ( 1-torch.sum(-N*-s1,0)**2 ) ) ).view(1,-1).repeat(3,1)*-N - 1/refra_index*-s1
point_cloud[0:2,:] = torch.cat( ( (( D - surface[2,:] ) * s2[0,:] / s2[2,:] + surface[0,:]).view(1,-1), \
                                  (( D - surface[2,:] ) * s2[1,:] / s2[2,:] + surface[1,:] ).view(1,-1)) , axis=0 )
point_cloud[2,:] = D

# Load the pre-computed cubic B-spline coefficients
all_W = torch.load('%s/all_W_%d.pt'%(folder,time_frame) )
all_W_dx = torch.load('%s/all_W_dx_%d.pt'%(folder,time_frame) )
all_W_dy = torch.load('%s/all_W_dy_%d.pt'%(folder,time_frame) )

# construct both point cloud and water surface
scale = param.scale
surface_normal_cuda = torch.zeros(3, xPixel*yPixel*time_frame).float().to(device)
image_cloud_cuda = image_cloud.float().to(device)
floMask_cuda = torch.from_numpy( floMask ).contiguous().float().view(1,-1).to(device)

all_W_cuda    = all_W.float().to(device)
all_W_dx_cuda = all_W_dx.float().to(device)
all_W_dy_cuda = all_W_dy.float().to(device)
camera = torch.from_numpy( param.camera ).float().to(device)

point_cloud_cuda = point_cloud.float().clone().to(device).requires_grad_(True)
height_map_cuda = (torch.ones( (time_frame, int(yPixel/scale), int(xPixel/scale) ))*param.water2Camera).float().to(device).requires_grad_(True)
model_pointCloudProj = pointCloudProj(param).float().to(device)
gd = Griddata_Cuda().float().to(device)

optimizer = torch.optim.Adam( [ {'params': point_cloud_cuda, 'lr': 5e-2}, \
                                {'params': height_map_cuda, 'lr': 1e-3} ] )

# boundaries are excluded when computing spatial losses
mask_cur = torch.ones(time_frame, 1, int(yPixel/scale), int(xPixel/scale)).float().to(device)
mask_cur[:, :, :4, :]  = 0
mask_cur[:, :, :, :4]  = 0
mask_cur[:, :, -4:, :] = 0
mask_cur[:, :, :, -4:] = 0

recon_surface = np.empty([0])
recon_point = np.empty([0])
start_time = time.time()
times = 0
for epoch in range(2000+1):
    if (epoch>=1000):
        optimizer.param_groups[1]['lr'] = 1e-4
    
    # store the estimated water surface and underwater point clouds every 50 iteration
    if (epoch)%50 == 0:
        point_cloud_com = point_cloud_cuda.detach().clone().cpu().numpy() 
        point_cloud_com = np.reshape(point_cloud_com, [3,yPixel,xPixel])
        point_cloud_com = np.transpose(point_cloud_com, [1, 2, 0])
        point_cloud_com = point_cloud_com[...,np.newaxis]
        if recon_point.shape[0]:
            recon_point = np.concatenate( (recon_point, point_cloud_com), axis = 3)
        else:    
            recon_point = point_cloud_com
            
    if (epoch)%50 == 0 and epoch<=2000:        
        surface_com = height_map_cuda.detach().clone().cpu().numpy() 
        surface_com = np.transpose(surface_com, [1, 2, 0])
        surface_com = surface_com[...,np.newaxis]
        if recon_surface.shape[0]:
            recon_surface = np.concatenate( (recon_surface, surface_com), axis = 3)
        else: 
            recon_surface = surface_com
    
    # distance loss
    D = model_pointCloudProj(point_cloud_cuda, height_map_cuda, image_cloud_cuda, \
                        surface_normal_cuda, all_W_cuda, all_W_dx_cuda, all_W_dy_cuda, floMask_cuda)
    
    # gradient loss
    point_cloud_com = point_cloud_cuda.view(3, yPixel, xPixel)

    u = point_cloud_com[0,:,:] / point_cloud_com[2,:,:] / param.pixel_size + (xPixel-1)/2
    v = point_cloud_com[1,:,:] / point_cloud_com[2,:,:] / param.pixel_size + (yPixel-1)/2
    uv = torch.cat( (u.view(yPixel,xPixel,1,1), v.view(yPixel,xPixel,1,1)), axis = 3 ).float()
    gd_depth = gd( point_cloud_com[2,:,:].view(yPixel,xPixel,1,1), uv ).view(1,1,yPixel,xPixel)
    mask_grad = torch.ones_like(gd_depth).float().to(device)
    mask_grad[gd_depth==0] = 0

    G_x = gradient_x( reflection_pad(gd_depth) )
    G_y = gradient_y( reflection_pad(gd_depth) )  
    G_x[abs(G_x)>param.water2Camera] = 0
    G_y[abs(G_y)>param.water2Camera] = 0
    gradient_loss = torch.sum( mask_grad * ( torch.abs(G_x) + torch.abs(G_y) ) )
    
    # curvature loss
    C_x = cur_x( reflection_pad(height_map_cuda.view(time_frame,1,int(yPixel/scale),int(xPixel/scale) )) )
    C_y = cur_y( reflection_pad(height_map_cuda.view(time_frame,1,int(yPixel/scale),int(xPixel/scale) )) )
    cur_loss = torch.sum(  mask_cur * ( torch.pow(C_x,2) + torch.pow(C_y,2) ) )
    
    # normal smoothness loss
    G_x = gradient_x( reflection_pad(height_map_cuda.view(time_frame,1,int(yPixel/scale),int(xPixel/scale) )) )
    G_y = gradient_y( reflection_pad(height_map_cuda.view(time_frame,1,int(yPixel/scale),int(xPixel/scale) )) )
    norm_loss = torch.sum(  mask_cur * ( torch.pow(G_x,2) + torch.pow(G_y,2) ) )
    
    # temporal loss
    temporal_loss = 0
    Laplacian = torch.squeeze(C_x + C_y)
    for i in range(1,time_frame-1):
        temporal_loss += torch.sum( mask_cur[i:i+1,0,:,:] * ((height_map_cuda[i+1,:,:] + height_map_cuda[i-1,:,:] - \
                2*height_map_cuda[i,:,:] - 0.5*Laplacian[i,:,:])**2 ) )
    
    kappa2 = 100
    kappa3 = 100
    kappa4 = 0.03
    loss = D + cur_loss*kappa2 + temporal_loss*kappa3 + gradient_loss*kappa4
  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch)%20 == 0:
        times = times + 1
        print ('%4d -- Total: %0.4f -- Distance: %0.4f -- Curvature: %0.4f -- Gradient: %0.4f -- Temporal: %0.4f' % \
              (epoch, loss, D, cur_loss, gradient_loss, temporal_loss) )

np.save('reconstruction/recon_surface.npy', recon_surface)            
np.save('reconstruction/recon_point.npy',   recon_point)    
print ("%0.4f seconds"%(time.time() - start_time) )