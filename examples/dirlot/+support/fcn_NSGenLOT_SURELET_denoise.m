function output = fcn_NSGenLOT_SURELET_denoise(input,nsgenlot,sigma)
%FCN_NSGENLOT_SURELET_DENOISE Removes additive Gaussian white noise using 
%   the inter-scale SURE-LET principle in the framework of an orthonormal
%   wavelet transform (OWT).
% 	
%   output = SUPPORT.FCN_NSGENLOT_SURELET_DENOISE(input,nsgenlot,sigma) 
%   performs an inter-scale orthonormal wavelet thresholding based on 
%   the principle exposed in [1].
%
%   Input:
%   - input : noisy signal of size [nx,ny].
%   - nsgenlot: Nonseparable GenLOT object provided by SaivDr Package
%   - (OPTIONAL) sigma : standard deviation of the additive Gaussian white
%   noise. By default 'sigma' is estimated using a robust median
%   estimator.
% 	
%   Output:
%   - output : denoised signal of the same size as 'input'.
%
%   See also fcn_min_sure, SaivDr Package
% 
%   Original Authors: Florian Luisier and Thierry Blu, March 2007
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%   This software is downloadable at http://bigwww.epfl.ch/
%
%   Modified by Shogo MURAMATSU for article [2], May 2012
%   Faculty of Engineering, Niigata University,
%   8050 2-no-cho Ikarashi, Nishi-ku, Niigata, 950-2181, JAPAN
%
%   References:
%   [1] F. Luisier, T. Blu, M. Unser, "A New SURE Approach to Image
%   Denoising: Interscale Orthonormal Wavelet Thresholding," 
%   IEEE Transactions on Image Processing, vol. 16, no. 3, pp. 593-606, 
%   March 2007.
%   [2] Shogo Muramatsu, "SURE-LET Image Denoising with Multiple DirLOTs,"
%   Proc. of 2012 Picture Coding Symposium (PCS2012), May 2012.
%
% http://msiplab.eng.niigata-u.ac.jp/
%
[nx,ny] = size(input);

% Compute the Most Suitable Number of Iterations
%-----------------------------------------------
J = aux_num_of_iters([nx,ny]);
nLevels = min(J);
fprintf(['nLevels = ' num2str(nLevels) '\n'])
if(nLevels==0)
    disp('The size of the signal is too small to perform a reliable denoising based on statistics.');
    output = input;
    return;
end

%% Analyzer and Synthesizer
%------------------------------
fnsgenlot = saivdr.dictionary.nsoltx.NsoltFactory.createAnalysis2dSystem(...
            nsgenlot,'BoundaryOperation','termination');
insgenlot = saivdr.dictionary.nsoltx.NsoltFactory.createSynthesis2dSystem(...
            nsgenlot,'BoundaryOperation','termination');
        
%% Orthonormal Wavelet Transform
%------------------------------
[coefs,scales] = step(fnsgenlot,input,nLevels);

%% Estimation of the Noise Standard Deviation 
%-------------------------------------------
if(nargin<3)
    if(nLevels>0)
        nHH = prod(scales(end-2,:));    
        HH  = coefs(end-3*nHH+1:end-2*nHH);
    end
    sigma = median(abs(HH(:)))/0.6745;
    fprintf(['\nEstimated Noise Standard Deviation: ' num2str(sigma,'%.2f') '\n']);
end

%% SURE-LET Denoising
%------------------
analysisFilters = step(nsgenlot,[],[]);
nOrientations = size(analysisFilters,3)-1;
w = cell(1,nOrientations);
for iOrientation=1:nOrientations
    h = analysisFilters(:,:,iOrientation+1);
    w{iOrientation} = h/norm(h(:));
end
%
nLL = prod(scales(1,:));
LL = reshape(coefs(1:nLL),scales(1,:));
sIdx = nLL+1;
iSubband = 2;
denoisedCoefs = coefs;
for iLevel = 1:nLevels
    for iOrientation=1:nOrientations
        eIdx = sIdx+prod(scales(iSubband,:))-1;
        Y  = reshape(coefs(sIdx:eIdx),scales(iSubband,:));
        Yp = abs(imfilter(LL,w{iOrientation},'conv','symmetric'));
        Yp = aux_gaussian_smoothing(Yp,1);
        subImg = fcn_min_sure(Y,Yp,sigma);
        denoisedCoefs(sIdx:eIdx) = subImg(:);
        %
        iSubband = iSubband + 1;
        sIdx = eIdx+1;
    end
    LL = step(insgenlot,coefs(1:eIdx),scales(1:iSubband-1,:));
end

%% Inverse transform
%------------------
output = step(insgenlot,denoisedCoefs,scales);
