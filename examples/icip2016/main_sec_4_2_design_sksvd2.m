%MAIN_SEC_4_2_DESIGN_SKSVD2 Dictionary Learning with 2-D Sparse K-SVD
%
% This script was used for the design of 2-D Sparse K-SVD 
% shown in Section 4.2. The results are stored under sub-folder `results.'
%
% The file name has the following form:
% 
%    "ksvds_b..._o..._n..._d..._(imagename).mat"
% 
% where the construction parameters are recognized by the rules
% 
%  * b... : Block size
%  * o... : ODCT size
%  * n... : # of coefficients
%  * d... : Atom sparsity
%
% The MAT file containts the following variables.
%
%  * Bsep : Base dictionary
%  * A    : Sparse dictionary representation matrix
%
% Reference
%    Ron Rubinstein,Michael Zibulevsky, and Michael Elad, "Double
%    sparsity: Learning sparse dictionaries for sparse signal 
%    approximatoin," IEEE Trans. Signal Process., vol. 58, no. 3,
%    pp.1553-1564, Mar. 2010.
%
% MATLAB functions in OMPS-Box v1 and KSVDS-Box v11 from
%
%    http://www.cs.technion.ac.il/~ronrubin/software.html
%
% were used.
%
% SVN identifier:
% $Id: main_pkg_design_sksvd2.m 875 2016-01-20 01:51:38Z sho $
%
% Requirements: MATLAB R2014a
%
% Copyright (c) 2014-2016, Shogo MURAMATSU
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

%% Setup KSVDSbox
support.fcn_setup_ksvdsbox

%% Set parameters
params.blocksize    = 4;     % 8x8
params.Tdict        = 3;     % sparsity of each trained atom
params.Tdata        = 2;     % number of coefficients
params.iternum      = 20;
params.trainnum     = 4096;  %
params.isMonitoring = false;
params.verbose      = 'irt';
params.dimension    = 2;     % 2D = 2 , 3D = 3
params.msgdelta     = 1;

%% Setup ODCT size 
% The followings calculate the redundancy of 2-D NSOLT
nDecs        = 16; % 2x2=4
nLevels      = 1;  % Tree levels of MS-NSOLT
nChs         = 24; % # of channels of MS-NSOLT 
redundancy = (nChs-1)*((nDecs^nLevels)-1)/((nDecs^nLevels)*(nDecs-1))+1/(nDecs^nLevels);
params.odctsize = ceil(params.blocksize*sqrt(redundancy));

%% Conditions  
imgSetLrn = { 'barbara512', 'lena512', 'goldhill512', 'baboon512' };   

%% Preperation
paramset = cell(length(imgSetLrn),1);
iPar = 1;
for iImg = 1:length(imgSetLrn)
    paramset{iPar} = params;
    paramset{iPar}.imgName  = imgSetLrn{iImg};
    paramset{iPar}.srcImg = im2double(support.fcn_load_testimg2(imgSetLrn{iImg}));
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
    [Bseps{iPar},As{iPar},~] = fcn_ksvdsdiclrn2(paramset{iPar});
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
