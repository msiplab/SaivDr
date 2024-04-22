function [h,angs,mus,resPsnr,resImg,feval,exitflag,output,grad,hessian] = ...
    fcn_eznsoltdiclrn2(srcImg,strnsolt,nIter,isnodc,stdinit)
%
% $Id: fcn_eznsoltdiclrn2.m 856 2015-11-02 02:01:16Z sho $
%
import saivdr.dictionary.generalfb.Analysis2dSystem
import saivdr.dictionary.generalfb.Synthesis2dSystem
import saivdr.sparserep.IterativeHardThresholding
import saivdr.utility.StepMonitoringSystem

if nargin < 3
    nIter = 10;
end

if nargin < 4
    isnodc = true;
end

if nargin < 5
    stdinit = 1e-2;
end

srcImg = im2double(srcImg);
nSparseCoefs = numel(srcImg)/4;

%% Initial parameters
[nsolt,angs,mus,nDec,nChs,nOrd] = setupFcn(strnsolt,isnodc);

%% Initialization
angs = angs + stdinit*randn(size(angs));
[h,f,~] = nsolt(angs);

%% Instantiation of step monitoring system for IHT
hf1 = figure(1);
sms = StepMonitoringSystem(...
    'SourceImage',srcImg,...
    'IsVisible',true,...
    'IsVerbose',true,...
    'IsPSNR',true,...
    'ImageFigureHandle',hf1);

%% Dictionary learning steps
options           = optimoptions(@fminunc);
options.PlotFcns  = @optimplotfval;
options.GradObj   = 'off';
options.Algorithm = 'quasi-newton';
for iter = 1:nIter
    
    % Instantiation of analyzser and synthesizer
    analyzer    = Analysis2dSystem(...
        'DecimationFactor',nDec,...
        'AnalysisFilters',h,...
        'FilterDomain','Frequency');
    synthesizer = Synthesis2dSystem(...
        'DecimationFactor',nDec,...
        'SynthesisFilters',f,...
        'FilterDomain','Frequency');
    
    % Instantiation of IHT for sparse approximation
    iht = IterativeHardThresholding(...
        'Synthesizer',synthesizer,...
        'AdjOfSynthesizer',analyzer,...
        'StepMonitor',sms);
    
    % Sparse Approximation
    [~,coefs,scales] = step(iht,srcImg,nSparseCoefs);
    
    % Dictionary Update
    cfcn = @(x) costFcn(nsolt,nDec,srcImg,coefs,scales,x);
    [angs,feval,exitflag,output,grad,hessian] = ...
        fminunc(cfcn,angs,options);
    
    % Result
    [h,f,E] = nsolt(angs);
    
end

%% Verification

% Assert size

assert(size(h,1)==nDec(1)*(nOrd(1)+1) || size(h,2)==nDec(2)*(nOrd(2)+1),...
    'Invalid size of analysis filters. (%dx%d)',size(h,1),size(h,2));
assert(size(f,1)==nDec(1)*(nOrd(1)+1) || size(f,2)==nDec(2)*(nOrd(2)+1),...
    'Invalid size of synthesis filters. (%dx%d)',size(f,1),size(f,2));

% Assert LP Condition

he = h(:,:,1:nChs(1));
df = he-flip(flip(he,1),2);
assert(norm(df(:)) < 1e-10,sprintf(...
    'Symmetric condition is violated (%g)',norm(df(:))));
ho = h(:,:,nChs(1)+1:end);
df = ho+flip(flip(ho,1),2);
assert(norm(df(:)) < 1e-10,sprintf(...
    'Antisymmetric condition is violated (%g)',norm(df(:))));

% Assert PU Condition
P  = double(E.'*E);
df = P(:,:,nOrd(1)+1,nOrd(2)+1) - eye(prod(nDec));
assert(norm(df(:)) < 1e-10,sprintf(...
    'Paraunitary condition is violated (%g)', norm(df(:))));

% Assert No-DC-leakage Condition
if isnodc
    for idx = 2:sum(nChs)
        H = h(:,:,idx);
        dc = abs(sum(H(:)));
        assert(dc < 1e-10,sprintf(...
            'No-DC-leakage condition is violated (%d:%g)',idx,dc));
    end
end

%% Output
for idx = 1:size(f,3)
    ps = nChs(1);
    px = max(nChs);
    if idx > ps
        subplot(2,px,idx-ps+px)
    else
        subplot(2,px,idx)
    end
    imshow(f(:,:,idx)+0.5)
end

if ~isnodc
    fprintf('\nDC-leakage is permitted.')
end
resImg  = step(iht,srcImg,nSparseCoefs);
resPsnr = psnr(im2uint8(srcImg),im2uint8(srcImg-resImg));
fprintf('\nPSNR: %f [dB]\n',resPsnr);

end

%% Functions
function cost = costFcn(nsolt,nDec,srcImg,coefs,scales,angs)
import saivdr.dictionary.generalfb.Synthesis2dSystem
[~,f,~] = nsolt(angs);
synthesizer = Synthesis2dSystem(...
    'DecimationFactor',nDec,...
    'SynthesisFilters',f,...
    'FilterDomain','Frequency');

recImg = step(synthesizer,coefs,scales);
difImg = srcImg-recImg;
cost = norm(difImg(:))^2;
end

function [nsolt,angs,mus,nDec,nChs,nOrd] = setupFcn(strnsolt,isnodc)

if strcmp(strnsolt,'d22c22o11')
    nDec = [ 2 2 ];
    nChs = [ 2 2 ];
    nOrd = [ 1 1 ];
    % W0=I, U0=I, Uy1=J, Ux1=-J
    mus  = [ 1 1  1 1  1 -1  -1 1 ].';
    angs = [ 0    0   -pi/2  -pi/2 ].';
    nsolt = @(x) eznsolt.fcn_type1d22c22(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c22o22')
    nDec = [ 2 2 ];
    nChs = [ 2 2 ];
    nOrd = [ 2 2 ];
    % W0=I, U0=I, Uy1=-I, Uy2=I, Ux1=-I, Ux2=I
    mus  = [ 1 1  1 1  -1 -1  1 1  -1 -1  1 1 ].';
    angs = [ 0 0 0 0 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type1d22c22(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c22o33')
    nDec = [ 2 2 ];
    nChs = [ 2 2 ];
    nOrd = [ 3 3 ];
    % W0=I, U0=I, U0y1=J, U0y2=-I, Uy3=-I, Ux1=I, Ux2=-I, Ux3=J
    mus  = [ 1 1  1 1  1 -1  -1 -1  -1 -1  1  1  -1 -1  1 -1 ].';
    angs = [ 0    0   -pi/2   0      0     0      0    -pi/2 ].';
    nsolt = @(x) eznsolt.fcn_type1d22c22(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o11')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 1 1 ];
    % W0=I, U0=I, Uy1=diag(J,1), Ux1= diag(-J,1),
    mus  = [ 1 1 1  1 1 1   1 -1 1    -1 1 1 ].';
    angs = [ 0 0 0  0 0 0  -pi/2 0 0  -pi/2 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type1d22c33(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o22')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 2 2 ];
    % W0=I, U0=I, Uy1=-I, Uy2=I, Ux1=-I, Ux2=I
    mus  = [ 1 1 1  1 1 1  -1 -1 -1  1 1 1  -1 -1 -1  1 1 1 ].';
    angs = zeros(18,1);
    nsolt = @(x) eznsolt.fcn_type1d22c33(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o33')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 3 3 ];
    % W0=I, U0=I, Uy1=diag(J,1), Uy2=-I, Uy3=-I, Uy1=I, Ux2=-I, Ux3=diag(J,1)
    mus  = [ 1 1 1   1 1 1    1 -1 1   -1 -1 -1  -1 -1 -1  1  1  1  -1 -1 -1   1  -1  1   ].';
    angs = [ 0 0 0   0 0 0   -pi/2 0 0  0  0  0   0  0  0  0  0  0   0  0  0  -pi/2 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type1d22c33(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c23o11')
    nDec = [ 2 2 ];
    nChs = [ 2 3 ];
    nOrd = [ 1 1 ];
    % Wi=I, Ui=I, Uiy1=J, U0=diag(-J,1)
    mus  = [ 1 1  1 1  1 -1  -1 1 1 ].';
    angs = [ 0    0   -pi/2  -pi/2 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type2d22c23o11(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c23o22')
    nDec = [ 2 2 ];
    nChs = [ 2 3 ];
    nOrd = [ 2 2 ];
    % W0=I, U0=I, Wy1=-I, Uy1=I, Wx1=-I, Ux1=I,
    mus  = [ 1 1  1 1 1  -1 -1  1 1 1  -1 -1  1 1 1 ].';
    angs = [ 0    0 0 0   0     0 0 0   0     0 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type2d22c23o22(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c23o33')
    nDec = [ 2 2 ];
    nChs = [ 2 3 ];
    nOrd = [ 3 3 ];
    % Wi=I, Ui=I, Uiy1=J, U0=diag(-J,1), Wy1=-I, Uy1=I, Wx1=-I, Ux1=I
    mus  = [ 1 1  1 1  1 -1  -1 1 1    -1 -1  1 1 1   -1 -1  1 1 1 ].';
    angs = [ 0    0   -pi/2  -pi/2 0 0  0      0 0 0   0     0 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type2d22c23o33(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c32o11')
    nDec = [ 2 2 ];
    nChs = [ 3 2 ];
    nOrd = [ 1 1 ];
    % Wi=-J, Ui=-J, Uiy1=J, W0=diag(-J,1)
    mus  = [ -1 1   -1 1    1 -1  -1 1 1 ].';
    angs = [ -pi/2  -pi/2  -pi/2  -pi/2 0 0 ].';
    nsolt = @(x) eznsolt.fcn_type2d22c32o11(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c32o22')
    nDec = [ 2 2 ];
    nChs = [ 3 2 ];
    nOrd = [ 2 2 ];
    % W0=I, U0=I, Wy1=I, Uy1=-I, Wx1=I, Ux1=-I,
    mus  = [ 1 1 1  1 1  1 1 1  -1 -1  1 1 1  -1 -1 ].';
    angs = [ 0 0 0  0    0 0 0   0     0 0 0   0    ].';
    nsolt = @(x) eznsolt.fcn_type2d22c32o22(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c32o33')
    nDec = [ 2 2 ];
    nChs = [ 3 2 ];
    nOrd = [ 3 3 ];
    % Wi=-J, Ui=-J, Uiy1=J, W0=diag(-J,1), Wy1=I, Uy1=-I, Wx1=I, Ux1=-I
    mus  = [ -1 1   -1 1    1 -1  -1 1 1     1 1 1  -1 -1  1 1 1  -1 -1 ].';
    angs = [ -pi/2  -pi/2  -pi/2  -pi/2 0 0  0 0 0   0     0 0 0   0    ].';
    nsolt = @(x) eznsolt.fcn_type2d22c32o33(x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o11sep')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 1 1 ];
    mus  = [ 1 1 -1   1 1 1 1 -1].';
    angs = 0;
    nsolt = @(x) eznsolt.fcn_sepd22c33(1,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o22sep')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 2 2 ];
    mus  = [ 1 1 -1 -1  -1 1 1 -1 -1 -1 ].';
    angs = [ 0 0 ];
    nsolt = @(x) eznsolt.fcn_sepd22c33(2,x,mus,isnodc);
elseif strcmp(strnsolt,'d22c33o33sep')
    nDec = [ 2 2 ];
    nChs = [ 3 3 ];
    nOrd = [ 3 3 ];
    mus  = [ 1 1 -1 -1 -1  1 1 -1 1 1 -1 1 1 ].';
    angs = [ 0 0 ];
    nsolt = @(x) eznsolt.fcn_sepd22c33(3,x,mus,isnodc);
else
    error('Not supported')
end

end