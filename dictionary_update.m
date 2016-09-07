setpath

srcImg = imread('peppers.png');
width  = 256; % Width
height = 256; % Height
px     = 64;  % Horizontal position of cropping
py     = 64;  % Vertical position of cropping
orgImg = im2double(srcImg(py:py+height-1,px:px+width-1,:));

obsImg = orgImg;

% Parameters for NSOLT
%nLevels = 4;     % # of wavelet tree levels
nLevels = 1;
nDec    = [2 2]; % Decimation factor
nChs    = 6; % # of channels
nOrd    = [2 2]; % Polyphase order
%nOrd = [0 0];
nVm     = 0;     % # of vanishing moments

%Parameter prepairation
V = eye(nChs);

Wx1 = dctmtx(nChs/2);
Wx2 = eye(nChs/2);
Wy1 = eye(nChs/2);
Wy2 = dctmtx(nChs/2);

Ux1 = -eye(nChs/2);
Ux2 = dctmtx(nChs/2);
Uy1 = -eye(nChs/2);
Uy2 = eye(nChs/2);

omfV = saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem();
omfW = saivdr.dictionary.utility.OrthonormalMatrixFactorizationSystem();

[angsV, musV] = step(omfV, V);

[angsWx1, musWx1] = step(omfW,Wx1);
[angsWx2, musWx2] = step(omfW,Wx2);
[angsWy1, musWy1] = step(omfW,Wy1);
[angsWy2, musWy2] = step(omfW,Wy2);
[angsUx1, musUx1] = step(omfW,Ux1);
[angsUx2, musUx2] = step(omfW,Ux2);
[angsUy1, musUy1] = step(omfW,Uy1);
[angsUy2, musUy2] = step(omfW,Uy2);

angsBx1 = pi/2;
angsBx2 = pi/2;
angsBy1 = pi/2;
angsBy2 = pi/2;

angs = [...
    angsV;...
    angsWx1;angsUx1;angsBx1;...
    angsWx2;angsUx2;angsBx2;...
    angsWy1;angsUy1;angsBy1;...
    angsWy2;angsUy2;angsBy2;...
    ];

mus = [...
    musV.';...
    musWx1.',musUx1.';...
    musWx2.',musUx2.';...
    musWy1.',musUy1.';...
    musWy2.',musUy2.';...
    ].';

nsolt = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
    'DecimationFactor',nDec,...
    'NumberOfChannels',nChs,...
    'PolyPhaseOrder', nOrd,...
    'NumberOfVanishingMoments',nVm);

set(nsolt,'Angles',angs);
set(nsolt,'Mus',mus);

%nsolt = tmp;

nItr = 10;
nCoefs = 10000;

for idx = 1:nItr
    analyzer = saivdr.dictionary.nsoltx.NsoltAnalysis2dSystem('LpPuFb2d',nsolt);
    synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
    
    iht = saivdr.sparserep.IterativeHardThresholding('Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
    [~,coefvec,scales] = step(iht,orgImg(:,:,1),nCoefs);
    
    preangs = get(nsolt,'Angles');
    
    options = optimoptions(@fminunc,...
        'Display','iter-detailed',...
        'Algorithm','quasi-newton',...
        'GradObj','off');
%        'MaxFunctionEvaluations',1000,...
    postangs = fminunc(@(xxx) hogehoge(orgImg(:,:,1),nsolt,xxx,coefvec,scales),preangs,options);
    
    set(nsolt,'Angles',postangs);
    atmimshow(nsolt);
end
