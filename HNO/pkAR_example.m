% load true image
Xbig = double(imread('peppers_gray.tif'));
Xbig = Xbig(:,:,1);
Xbig = (Xbig - min(min(Xbig))) / (max(max(Xbig)) - min(min(Xbig)));

% PSF
c = 21;
center = [c, c];
P = psfGauss([2*c-1,2*c-1], 4);

% Blurring
Pbig = padPSF(P, size(Xbig));
Sbig = fft2(circshift(Pbig, 1-c)); 
Bbig = real(ifft2(Sbig .* fft2(Xbig)));  
X = Xbig(c:end-c,c:end-c);
B = Bbig(c:end-c,c:end-c);

% 1% of Gaussian noise
E = randn(size(B));
E = E / norm(E,'fro');
B = B + 0.01*norm(B,'fro')*E;
PSF = padPSF(P, size(B));

% restoration choosing lambda by GCV
Xnorm = norm(X,'fro');
ar_X = tik_AR2(B, PSF, center);
ar_err = norm(ar_X-X)/Xnorm;
re_X = tik_dct(B, PSF, center);
re_err = norm(re_X-X)/Xnorm;

% show the images
figure
subplot(2,2,1), imshow(X), title('True')
subplot(2,2,2), imshow(B), title('Observed')
subplot(2,2,3), imshow(ar_X), title(['Anti-refl: ' num2str(ar_err)])
subplot(2,2,4), imshow(re_X), title(['Reflect: ' num2str(re_err)])
