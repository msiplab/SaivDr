function fcn_sweeplambda(params)
%FCN_SWEEPLAMBDA Evaluation of restoration with parameter lambda sweep
%
% fcn_sweeplambda(orgImg,obsImg,strpartpic,rstrset,strdics) evaluate
% image restoration performance by sweeping lambda, which controls
% the weight between the fidelity and sparsity term.
%
% SVN identifier:
% $Id: fcn_sweeplambda.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
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
orgImg       = params.orgImgSet;
obsImg       = params.obsimgSet;
strpartpic   = params.strImgSet;
rstrset      = params.imrstrSet;
stralg       = params.stralg;
strdics      = params.strDicSet;
strlinproc   = params.strlinproc;
nImgs        = params.nImgs;
nDics        = params.nDics;
nLambdas     = params.nLambdas;
stepLambda   = params.stepLambda;
offsetLambda = params.offsetLambda;
stepmonitor  = params.stepmonitor;
nsigma       = params.nsigma;
%
for iImg = 1:nImgs
    orgImg_ = orgImg{iImg};
    obsImg_ = obsImg{iImg};
    strpartpic_ = strpartpic{iImg};
    for iDic = 1:nDics
        psnrswp = zeros(1,nLambdas);
        ssimswp = zeros(1,nLambdas);
        rstr   = rstrset{iDic};
        strdic = strdics{iDic};
        parfor iLambda = 1:nLambdas
            lambda_ = (iLambda-1)*stepLambda + offsetLambda;
            rstr_ = clone(rstr);
            stepmonitor_ = clone(stepmonitor);
            set(stepmonitor_,'SourceImage',orgImg_);
            set(rstr_,'StepMonitor',stepmonitor_);
            set(rstr_,'Lambda',lambda_);
            % Restoration
            step(rstr_,obsImg_);
            % Extract the final result
            nItr   = get(stepmonitor_,'nItr');
            psnrs_ = get(stepmonitor_,'PSNRs');
            mses_  = get(stepmonitor_,'MSEs');            
            ssims_ = get(stepmonitor_,'SSIMs');
            % Extract the final values
            psnrswp(iLambda) = psnrs_(nItr);
            mseswp(iLambda)  = mses_(nItr);  %#ok
            ssimswp(iLambda) = ssims_(nItr);
            fprintf('(%s) lambda = %7.4f : nItr = %4d, psnr = %6.2f [dB], ssim = %6.3f\n',...
                strdic, lambda_, nItr, psnrs_(nItr), ssims_(nItr));
        end
        
       %% Extract the best result
        lambda = (0:nLambdas-1)*stepLambda + offsetLambda;
        % Search Max. values
        [maxpsnr,idxpsnrmax] = max(psnrswp(:));
        [maxssim,idxssimmax] = max(ssimswp(:)); %#ok
        lambdamaxpsnr = lambda(idxpsnrmax);
        lambdamaxssim = lambda(idxssimmax);     %#ok
        % Update the restoration system by the best value of lambda
        set(rstr,'Lambda',lambdamaxpsnr);
        % String for identifing the condtion
        scnd = sprintf('%s_%s_%s_%s_ns%06.2f',...
            strpartpic_,lower(stralg),lower(strdic),strlinproc,nsigma);
        % File name
        psnrsname = sprintf('./results/psnr_ssim_%s.mat',scnd);
        disp(psnrsname)
        % Save data
        save(psnrsname,'lambda','psnrswp','mseswp','ssimswp','rstr',...
            'maxpsnr','lambdamaxpsnr','maxssim','lambdamaxssim')
        fprintf('*(%s) lambda = %7.5f : max psnr = %6.2f [dB]\n',...
            strdic, lambdamaxpsnr, maxpsnr);
    end
end
