%MAIN_PARSKSVDSDICLRN Spars-KSVD dictionary learning
%
% This script executes Sparse-KSVD dictionary learning
%
% Reference
%    Ron Rubinstein,Michael Zibulevsky, andMichael Elad, "ouble
%    sparsity: Learning sparse dictionaries for sparse signal 
%    approximatoin," IEEE Trans. Signal Process., vol. 58, no. 3,
%    pp.1553?1564, Mar. 2010.
%
% MATLAB functions in OMPS-Box v1 and KSVDS-Box v11 from
%
%    http://www.cs.technion.ac.il/?ronrubin/software.html
%
% were used.
%
% SVN identifier:
% $Id: main_udhaarimsr.m 146 2014-01-18 06:55:25Z sho $
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

%% Setup KSVDSbox
support.fcn_setup_ksvdsbox

%% Set parameters
params.blocksize = 8; % 8x8
params.Tdict = 6;                      % sparsity of each trained atom
params.Tdata = 8;                      % number of coefficients
params.iternum = 15;
params.trainnum = 10000;
params.isMonitoring = false;
params.verbose = 'irt';

%% Setup ODCT size 
nDecs   = 4;
nLevels = 7; %
nChs    = 8; % 4+4
redundancy = (nChs-1)*((nDecs^nLevels)-1)/((nDecs^nLevels)*(nDecs-1))+1/(nDecs^nLevels);
params.odctsize = ceil(params.blocksize*sqrt(redundancy));

%% Conditions
imgSet = { 'barbara128', 'lena128', 'goldhill128', 'baboon128' };   

%% Preperation
paramset = cell(length(imgSet),1);
iPar = 1;
for iImg = 1:length(imgSet)
    paramset{iPar} = params;
    paramset{iPar}.imgName  = imgSet{iImg};
    paramset{iPar}.srcImg = im2double(support.fcn_load_testimg(imgSet{iImg}));
    iPar = iPar + 1;
end

%% Dictionary learning
nPars = length(paramset);
Bseps = cell(nPars,1);
As = cell(nPars,1);
fnames = cell(nPars,1);
parfor iPar = 1:nPars
    fprintf('Dictionary learning condition: b%d o%d n%d d%d %s...\n',...
        paramset{iPar}.blocksize,paramset{iPar}.odctsize,...
        paramset{iPar}.Tdata,paramset{iPar}.Tdict,...
        paramset{iPar}.imgName);
    [Bseps{iPar},As{iPar},~] = fcn_ksvdsdiclrn(paramset{iPar});
    fnames{iPar} = sprintf('./results/ksvds_b%d_o%d_n%d_d%d_%s',...
        paramset{iPar}.blocksize,paramset{iPar}.odctsize,...
        paramset{iPar}.Tdata,paramset{iPar}.Tdict,...
        paramset{iPar}.imgName);
end

%% Save data
for iPar = 1:nPars
    Bsep = Bseps{iPar};
    A = As{iPar};
    disp(fnames{iPar})
    save(fnames{iPar},'Bsep','A')
end
