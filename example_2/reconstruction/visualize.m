clear; close all
addpath('tools');

%% Notice that the visulization is not aligned with the input video, a simple flip and rotation can match them

%% visualize point clouds
img = im2double( imread('../parameters/ref.png') );
% folder = 'recon_point.npy';
folder = 'example_point.npy';

pointCloud = readNPY(folder);

xPixel = size(pointCloud,2);
yPixel = size(pointCloud,1);
point = -pointCloud(:,:,:,end); % change index to see the reconstruction at various time steps
b = 10;
gap = 1;
fig = figure; set(fig, 'Position',[500 100 750 400]);
scatter3( reshape( point(b:gap:end-b,b:gap:end-b,1), [] ,1 ), ...
    reshape( point(b:gap:end-b,b:gap:end-b,2), [] ,1 ),...
    reshape( point(b:gap:end-b,b:gap:end-b,3), [] ,1 ), 40, ...
    reshape( img(b:gap:end-b,b:gap:end-b,:), [] ,3 ) , 'filled' ); 
view(185,50);
zlim([-50 -30]);
set(gca,'xtick',[]); set(gca,'ytick',[]); set(gca,'ztick',[]); box off; axis off;


%% visualize dynamic water surfaces
folder = 'recon_surface.npy';
% folder = 'example_surface.npy';    % example_surface.npy is large, available upon request
reSurface = 20 - readNPY(folder);

scale = 2;  % make the scale value consistent with the reconstruction code
[x,y] = meshgrid( 1:xPixel/scale, 1:yPixel/scale );

h = figure; set(gcf,'Position',[1000 1000 1500 1000]);
filename = ['moving_surface.gif'];
for t = 1:size(reSurface,3)
    
    z = reSurface(:,:,t,end);
    surf( x(b:end-b,b:end-b), y(b:end-b,b:end-b), z(b:end-b,b:end-b) );
    shading interp
    axis([ 0 xPixel/scale 0 yPixel/scale -0.3 0.3]);
    set(gca,'xtick',[]); set(gca,'ytick',[]); set(gca,'ztick',[]); box off; axis off;
    view(-40,30); caxis([-0.3 0.3])
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    if t == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end
end