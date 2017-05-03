% MAIN_HSSOSHRINKAGE: Shrinkage Denoising with Wavelets
%
% SVN identifier:
% $Id: main_hssoshrinkage.m 749 2015-09-02 07:58:45Z sho $
%
% References:
%   Shogo Muramatsu:
%   ''SURE-LET Image Denoising with Multiple DirLOTs,''
%   Proc. of 2012 Picture Coding Symposium (PCS2012), May 2012.
%
% Requirements: MATLAB R2014a
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
close all; clc

%% Download OWT_SURELET

if ~exist('OWT_SURELET','dir')
    fprintf('Downloading OWT_SURELET.zip.\n');
    url = 'http://bigwww.epfl.ch/demo/suredenoising/matlab/OWT_SURELET.zip';
    unzip(url,'./OWT_SURELET')
    fprintf('Done!\n');
end

addpath('./OWT_SURELET/OWT SURE-LET')

%% Read test images

imgSet = { 'goldhill' 'lena' 'baboon' 'barbara' };
srcImg = cell(length(imgSet),1);
for iImg = 1:length(imgSet)
    srcImg{iImg} = support.fcn_load_testimg2(imgSet{iImg});
end

%{
for iImg = 1:length(srcImg)
    subplot(1,2,iImg), subimage(srcImg{iImg});
    axis off;
end
drawnow
%}

%%
sigma = [10 20 30 40 50];
wtype = { 'sym5' 'son4' 'udn4' };
%
shfcn{1} = @(x) OWT_SURELET_denoise(x,wtype{1});
%
S = load('./filters/nsgenlot_d2x2_o4+4_v2.mat','lppufb');
shfcn{2} = @(x) support.fcn_NSGenLOT_SURELET_denoise(x,S.lppufb);
%
shfcn{3} = @(x) support.fcn_HSDirLOT_SURELET_denoise(x);

%%
for iImg = 1:length(srcImg)
    src = srcImg{iImg};
    p = im2double(src);
    psnrtab = zeros(length(sigma),length(wtype));
    ssimtab = zeros(length(sigma),length(wtype));
    msetab  = zeros(length(sigma),length(wtype));
    for iSigma = 1:length(sigma)
        ssigma = sigma(iSigma);
        v = (ssigma/255)^2;
        strpic = imgSet{iImg};
        fnoisy = sprintf('./images/noisy_%s_%d_0.tif',...
            strpic,ssigma);
        if exist(fnoisy,'file')
            fprintf('\n')
            disp(['Read ' fnoisy])
            b = im2double(imread(fnoisy));
        else
            fprintf('\n')
            disp(['Generate ' fnoisy])
            b = imnoise(p,'gaussian',0,v);
            imwrite(im2uint8(b),fnoisy);
        end
        cpsnr = cell(1,length(shfcn));
        cssim = cell(1,length(shfcn));
        cmse  = cell(1,length(shfcn));
        fprintf('\n%s %d\n',strpic,ssigma)
        for iFcn=1:length(shfcn)
            y = shfcn{iFcn}(b);
            cpsnr{iFcn} = psnr(p,y);
            cmse{iFcn}  = (norm(p(:)-y(:))^2)/numel(p);
            res = im2uint8(y);
            cssim{iFcn} = ssim(src,res);
            disp(['(' wtype{iFcn} ') mse= ' num2str(cmse{iFcn}*255^2) ...
                ', psnr=' num2str(cpsnr{iFcn}) ' [dB]' ...
                ', ssim=' num2str(cssim{iFcn}) ]);
            imwrite(res,sprintf('./images/result_%s_%s_%d.tif',...
                wtype{iFcn},strpic,ssigma));
        end
        for iFcn=1:length(shfcn)
            psnrtab(iSigma,iFcn) = cpsnr{iFcn};
            msetab(iSigma,iFcn)  = cmse{iFcn};
            ssimtab(iSigma,iFcn) = cssim{iFcn};
        end
        for iTrial = 1:4
            fnoisy = sprintf('./images/noisy_%s_%d_%d.tif',...
                strpic,ssigma,iTrial);
            if exist(fnoisy,'file')
                fprintf('\n')
                disp(['Read ' fnoisy])
                b = im2double(imread(fnoisy));
            else
                fprintf('\n')
                disp(['Generate ' fnoisy])
                b = imnoise(p,'gaussian',0,v);
                imwrite(im2uint8(b),fnoisy);
            end
            fprintf('\n%s %d\n',strpic,ssigma)
            parfor iFcn=1:length(shfcn)
                y = shfcn{iFcn}(b);
                cpsnr{iFcn} = psnr(p,y);
                cmse{iFcn}  = (norm(p(:)-y(:))^2)/numel(p);
                res = im2uint8(y);
                cssim{iFcn} = ssim(src,res);
                disp(['(' wtype{iFcn} ') mse= ' num2str(cmse{iFcn}*255^2) ...
                    ', psnr=' num2str(cpsnr{iFcn}) ' [dB]' ...
                    ', ssim=' num2str(cssim{iFcn}) ]);
            end
            for iFcn=1:length(shfcn)
                psnrtab(iSigma,iFcn) = psnrtab(iSigma,iFcn) + ...
                    (cpsnr{iFcn} - psnrtab(iSigma,iFcn) ) / (iTrial + 1 );
                ssimtab(iSigma,iFcn) = ssimtab(iSigma,iFcn) + ...
                    (cssim{iFcn} - ssimtab(iSigma,iFcn) ) / (iTrial + 1 );
                msetab(iSigma,iFcn) = msetab(iSigma,iFcn) + ...
                    (cmse{iFcn} - msetab(iSigma,iFcn) ) / (iTrial + 1 );
            end
        end
        disp(psnrtab)
        disp(ssimtab)
        disp(msetab)
        save(sprintf('./results/surelet_psnrs_%s',imgSet{iImg}),...
            'sigma','wtype','psnrtab','msetab','ssimtab')
    end
end
