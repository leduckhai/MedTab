%%   Distance Regularized Level Set Evolution Segmentation Function (2 matrix detectors)
%   _______________________________________________________________________
%   Le Duc Khai
%   Bachelor in Biomedical Engineering
%   FH Aachen - University of Applied Sciences, Germany.
%
%   Last updated on 15.02.2019.
%
%   The proposed algorithm creates active contour based on Level Set Evolution 
%   principles without re-initialization step needed.
%   This function works well with images which contain many small details.
%   Its results' accuracy is heavily based on input parameters.
%
%   Implementation is based on this scientific paper:
%       Chunming Li, Chenyang Xu, Changfeng Gui, and Martin D. Fox
%       "Distance Regularized Level Set Evolution and Its Application to Image Segmentation"
%       IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 19, NO. 12, DECEMBER 2010
%
%   The following codes are implemented only for PERSONAL USE, e.g improving
%   programming skills in the domain of Image Processing and Computer Vision.
%   If you use this algorithm, please cite the paper mentioned above to support
%   the authors.
%
%   Parameters:
%       image: the input image
%       timestep: the time step
%       mu: coefficient of the distance regularization term R(phi)
%       lambda: coefficient of the weighted length term L(phi)
%       alpha: coefficient of the weighted area term A(phi)
%       epsilon: width of Dirac Delta function
%       maxiter: number of maximum iterations
%       sigma: standard deviation of Gaussian distribution
%       state: detector dilates or constricts
%       m1: width of the initial phi matrix (syntax example: 150:170)
%       n1: height of the initial phi matrix (syntax example: 200:220)
%       m2
%       n2
%
%   Examples:
%       timestep: 1 ~ 5, the higher the time step, the lesser maxiter should be
%       mu = 0.2
%       lambda = 5
%       alpha = -3
%       epsilon = 1.5
%       maxiter: do some experiments to find the best value at which
%                contour stops perfectly
%       sigma: 1 ~ 5
%       state: 'dilate' or 'constrict'
%       m1: adjust so that objects are inside the rectangle
%       n1: adjust so that objects are inside the rectangle
%       m2
%       n2
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function DRLSE_segment_2m(image, timestep, mu, lambda, alpha, epsilon, maxiter, sigma, state, m1, n1, m2, n2)
%% Read the inputs
if ischar(image) == 1
    image = imread(image);
    original_image = image;
else 
    original_image = image;
end
if ndims(image) == 3
    image = rgb2gray(original_image);
end
figure(1); 
subplot(3,2,1); imshow(original_image); title('Original image');
subplot(3,2,2); imshow(image); title('Grayscale image');

%% Apply Gaussian Filter to smooth the image
image_smooth = double(imgaussfilt(image, sigma));
subplot(3,2,3); imshow(image_smooth, []); title('Gaussian-smoothed image');

%% Edge indicator (Equation 23)
[dx dy]=gradient(image_smooth);
g = 1./(1 + (dx.^2 + dy.^2));  % Edge indicator function
subplot(3,2,4); imshow(g); title('Edge indicator g');

%% Initial phi
c0 = 2;
switch state 
    case 'constrict'
        phi = -c0*ones(size(image));
        phi(m1, n1) = c0;
        phi(m2, n2) = c0;
    case 'dilate'
        phi = c0*ones(size(image));
        phi(m1, n1) = -c0;
        phi(m2, n2) = -c0;
end
subplot(3,2,5); imshow(phi); title('Initial phi matrix');

%% Main loop of phi
[vx vy] = gradient(g);
frm = 0;
for k = 1:maxiter
    % Check boundary conditions
    phi = NeumannBoundCond(phi);
    
    % Calculate differential of regularized term in Equation 30
    distRegTerm = distReg_p2(phi);
    
    % Calculate differential of area term in Equation 30
    diracPhi = Dirac(phi, epsilon);
    areaTerm = diracPhi.*g;
    
    % Calculate differential of length term in Equation 30
    [phi_x phi_y] = gradient(phi);
    s = sqrt(phi_x.^2 + phi_y.^2);
    Nx = phi_x./(s + 1e-10); % add a small positive number to avoid division by zero
    Ny = phi_y./(s + 1e-10);
    edgeTerm = diracPhi.*(vx.*Nx + vy.*Ny) + diracPhi.*g.*div(Nx,Ny);
    
    % Update phi according to Equation 20
    phi = phi + timestep*(mu/timestep*distRegTerm + lambda*edgeTerm + alpha*areaTerm);
    
    % Show result in every 50 iteration
    if mod(k, 50) == 1
        frm = frm + 1;
        h = figure(2);
        set(gcf, 'color', 'w');
        subplot(1,2,1);
        II = image;
        imshow(II); axis off; axis equal; hold on;  
        q = contour(phi, [0,0], 'r');
        msg = ['contour result , iteration number=' num2str(k)];
        title(msg);
        subplot(1,2,2);
        mesh(-phi); 
        hold on;  contour(phi, [0,0], 'r','LineWidth',2);
        
        view([-55+180 55]);      
        msg=['phi result , iteration number=' num2str(k)];
        title(msg);
        pause(0.1)
        
        % Gif video        
        frame = getframe(h);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256); 
    end 
end

%% Show last iteration results
figure(3);
imagesc(image,[0 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
msg=['phi result , iteration number=' num2str(k)];
title(msg);

%% Component functions
function f = distReg_p2(phi)
% Compute the distance regularization term with the double-well potential p2 in equation 16
[phi_x phi_y]=gradient(phi);
s = sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps = a.*sin(2*pi*s)/(2*pi) + b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps.*phi_x - phi_x, dps.*phi_y - phi_y) + 4*del2(phi);
end

function f = div(nx,ny)
[nxx junk]=gradient(nx);
[junk nyy]=gradient(ny);
f = nxx + nyy;
end

function f = Dirac(x, sigma)
f = (1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;
end

function g = NeumannBoundCond(f)
[rows cols] = size(f);
g = f;
g([1 rows],[1 cols]) = g([3 rows-2],[3 cols-2]);
g([1 rows],2:end-1) = g([3 rows-2],2:end-1);
g(2:end-1,[1 cols]) = g(2:end-1,[3 cols-2]);
end

end

