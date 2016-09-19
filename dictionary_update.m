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

nsolt = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
    'DecimationFactor',nDec,...
    'NumberOfChannels',nChs,...
    'PolyPhaseOrder', nOrd,...
    'NumberOfVanishingMoments',nVm);

nItr = 3;
nCoefs = 30000;

for idx = 1:nItr
    analyzer = saivdr.dictionary.nsoltx.NsoltAnalysis2dSystem('LpPuFb2d',nsolt);
    synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
    
    iht = saivdr.sparserep.IterativeHardThresholding('Synthesizer',synthesizer,'AdjOfSynthesizer',analyzer);
    [~,coefvec,scales] = step(iht,orgImg(:,:,1),nCoefs);
    
    preangs = get(nsolt,'Angles');
    
    options = optimoptions(@fminunc,...
        'Display','iter-detailed',...
        'Algorithm','quasi-newton',...
        'GradObj','off',...
        'MaxFunctionEvaluations',10000);
    postangs = fminunc(@(xxx) hogehoge(orgImg(:,:,1),nsolt,xxx,coefvec,scales),preangs,options);
    
    set(nsolt,'Angles',postangs);
    atmimshow(nsolt);
end
