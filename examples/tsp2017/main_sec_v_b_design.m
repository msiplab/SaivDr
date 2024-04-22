%MAIN_SEC_V_B_DESIGN Example-based design of 3-D NSOLT (Fig. 9)
%
% This script was used for the design of 3-D NSOLTs in Section V.B.
% The results are stored under sub-folder `results.'
% 
% The following materials are also reproduced:
%
% - tiff/fig09.tif
% - materials/fig09.tex
%
% SVN identifier:
% $Id: main_sec_v_b_design.m 852 2015-10-29 21:42:00Z sho $
%
% Requirements: MATLAB R2014a
%
%  * Signal Processing Toolbox
%  * Image Processing Toolbox
%  * Optimization Toolbox
%
% Recommended:
% 
%  * MATLAB Coder
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
% LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
%
nfig   = 9;     % Figure number      
isEdit = false; % Edit mode

setpath

%% Preparation

rng(0,'twister') % Initalization of random generator
nTrials = 20;   % # of trials
stdinit = pi/6; % Standard deviation for random initialization
nIter   = 10;   % # of iterations
isnodc  = true; % No-DC-leakage condition

%% Load training image
imgName = 'mrbrain';
strImg  = 'mrbrain32x32x32';               
srcImg  = support.fcn_load_testimg3(strImg);

%% Parameters
% d: Downsampling factor (M0, M1 and M2)
% c: # of channels (ps and pa)
% o: # of polyphase order (N0, N1 and N2)
nsoltset = {...
    'd222c44o222',...
    'd222c45o222',...
    'd222c54o222',...
    'd222c55o222'...
    };

%% Example-based design process
if ~isEdit
    for insolt = 1:length(nsoltset)
        strnsolt = nsoltset{insolt};
        h_       = cell(nTrials,1);
        angs_    = cell(nTrials,1);
        mus_     = cell(nTrials,1);
        resPsnr_ = cell(nTrials,1);
        output_  = cell(nTrials,1);
        parfor iRep = 1:nTrials
            [h,angs,mus,resPsnr,~,~,~,output] = ...
                eznsolt.fcn_eznsoltdiclrn3(...
                srcImg,strnsolt,nIter,isnodc,stdinit);
            h_{iRep}       = h;
            angs_{iRep}    = angs;
            mus_{iRep}     = mus;
            resPsnr_{iRep} = resPsnr;
            output_{iRep}  = output;
        end
        %
        fname = sprintf('results/eznsolt_%s_ndc%d_%s',...
            strnsolt,isnodc,strImg);
        if exist([fname '.mat'],'file') == 2
            s = load(fname,'resPsnr');
            prePsnr = s.resPsnr;
        else
            prePsnr = 0;
        end
        %
        for iRep = 1:nTrials        
            h = h_{iRep};
            angs = angs_{iRep};
            mus = mus_{iRep};
            resPsnr = resPsnr_{iRep};
            output = output_{iRep};
            if resPsnr > prePsnr
                save(fname,...
                    'h','angs','mus','resPsnr','nIter','isnodc',...
                    'output')
                prePsnr = resPsnr;                
                fprintf('Updated! %f -> %f\n', prePsnr, resPsnr)
            else
                fprintf('Not updated. %f < %f\n', resPsnr, prePsnr)
            end
        end
    end
end

%% Store the training image
[nRows, nCols, nLays ] = size(srcImg);
srcImg = padarray(srcImg,[1 1],1,'both');
hm = montage(permute(srcImg,[1 2 4 3]),'Size',[4 8]);
imwrite(get(hm,'CData'),sprintf('tiff/fig%02d.tif',nfig))

%% Produce LaTeX file
sw = StringWriter();
sw.addcr('%#! latex double')
sw.addcr('%')
sw.addcr('% $Id: main_sec_v_b_design.m 852 2015-10-29 21:42:00Z sho $')
sw.addcr('%')
sw.addcr('\begin{figure}[tb]')
sw.addcr('\centering')
sw.add('\centerline{\includegraphics[width=60mm]{')
sw.add(sprintf('fig%02d',nfig))
sw.addcr('}}')
sw.add('\caption{Training volume data, part of {\it ')
sw.add(imgName)
sw.add('}, that is ')
sw.add(sprintf('$%d\\times %d\\times %d$',nRows,nCols,nLays))
sw.addcr(' voxels in 12-bpp grayscale.}')
sw.addcr(sprintf('\\label{fig:%02d}',nfig))
sw.addcr('\end{figure}')
sw.add('\endinput')

disp(sw)
write(sw,sprintf('materials/fig%02d.tex',nfig))