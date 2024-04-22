function [h,angs,mus,resPsnr,resImg,feval,exitflag,output,grad,hessian] = ...
    fcn_eznsoltdiclrn3(srcImg,strnsolt,nIter,isnodc,stdinit)
%
% $Id: fcn_eznsoltdiclrn3.m 856 2015-11-02 02:01:16Z sho $
%
import saivdr.dictionary.generalfb.Analysis3dSystem
import saivdr.dictionary.generalfb.Synthesis3dSystem
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
    
    % Instantiation of analyzer and synthesizer
    analyzer    = Analysis3dSystem(...
        'DecimationFactor',nDec,...
        'AnalysisFilters',h,...
        'FilterDomain','Frequency');
    synthesizer = Synthesis3dSystem(...
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
assert(size(h,1)==nDec(1)*(nOrd(1)+1) || size(h,2)==nDec(2)*(nOrd(2)+1) || size(h,3)==nDec(3)*(nOrd(3)+1),...
    'Invalid size of analysis filters. (%dx%dx%d)',size(h,1),size(h,2),size(h,3));
assert(size(f,1)==nDec(1)*(nOrd(1)+1) || size(f,2)==nDec(2)*(nOrd(2)+1) || size(f,3)==nDec(3)*(nOrd(3)+1),...
    'Invalid size of synthesis filters. (%dx%dx%d)',size(f,1),size(f,2),size(f,3));

% Assert LP Condition

he = h(:,:,:,1:nChs(1));
df = he-flip(flip(flip(he,1),2),3);
assert(norm(df(:)) < 1e-10,sprintf(...
    'Symmetric condition is violated (%g)',norm(df(:))));
ho = h(:,:,:,nChs(1)+1:end);
df = ho+flip(flip(flip(ho,1),2),3);
assert(norm(df(:)) < 1e-10,sprintf(...
    'Antisymmetric condition is violated (%g)',norm(df(:))));

% Assert PU Condition
P  = double(E.'*E);
df = P(:,:,nOrd(1)+1,nOrd(2)+1,nOrd(3)+1) - eye(prod(nDec));
assert(norm(df(:)) < 1e-10,sprintf(...
    'Paraunitary condition is violated (%g)', norm(df(:))));

% Assert No-DC-leakage Condition
if isnodc
    for idx = 2:sum(nChs)
        H = h(:,:,:,idx);
        dc = abs(sum(H(:)));
        assert(dc < 1e-10,sprintf(...
            'No-DC-leakage condition is violated (%d:%g)',idx,dc));
    end
end

%% Output
if ~isnodc
    fprintf('\nDC-leakage is permitted.')
end
resImg  = step(iht,srcImg,nSparseCoefs);
resPsnr = psnr(srcImg(:),srcImg(:)-resImg(:));
fprintf('\nPSNR: %f [dB]\n',resPsnr);

end

%% Cost function
function cost = costFcn(nsolt,nDec,srcImg,coefs,scales,angs)
import saivdr.dictionary.generalfb.Synthesis3dSystem
[~,f,~] = nsolt(angs);
synthesizer = Synthesis3dSystem(...
    'DecimationFactor',nDec,...
    'SynthesisFilters',f,...
    'FilterDomain','Frequency');

recImg = step(synthesizer,coefs,scales);
difImg = srcImg-recImg;
cost = norm(difImg(:))^2;
end

function [nsolt,angs,mus,nDec,nChs,nOrd] = setupFcn(strnsolt,isnodc)

if strcmp(strnsolt,'d222c44o222')
    nDec = [ 2 2 2 ];
    nChs = [ 4 4 ];
    nOrd = [ 2 2 2 ];
    mus  = [  ones(4,1) ;
        -ones(4,1) ;
        -ones(4,1) ;
        ones(4,1) ;
        -ones(4,1) ;
        ones(4,1) ;
        -ones(4,1) ;
        ones(4,1) ];
    angs = zeros(6*8,1);
    nsolt = @(x) eznsolt.fcn_type1d222c44(nOrd,x,mus,isnodc);
elseif strcmp(strnsolt,'d222c45o222')
    nDec = [ 2 2 2 ];
    nChs = [ 4 5 ];
    nOrd = [ 2 2 2 ];
    mus  = repmat([-ones(nChs(1),1); ones(nChs(2),1)],[4 1]);
    angs = zeros(16*4,1);
    nsolt = @(x) eznsolt.fcn_type2d222c45o222(x,mus,isnodc);
elseif strcmp(strnsolt,'d222c54o222')
    nDec = [ 2 2 2 ];
    nChs = [ 5 4 ];
    nOrd = [ 2 2 2 ];
    mus  = repmat([ones(nChs(1),1); -ones(nChs(2),1)],[4 1]);
    angs = zeros(16*4,1);
    nsolt = @(x) eznsolt.fcn_type2d222c54o222(x,mus,isnodc);
elseif strcmp(strnsolt,'d222c55o222')
    nDec = [ 2 2 2 ];
    nChs = [ 5 5 ];
    nOrd = [ 2 2 2 ];
    mus  = [ ones(nChs(1),1) ; -ones(nChs(2),1) ;
        repmat( [ -ones(nChs(1),1) ; ones(nChs(2),1)],[3 1])];
    angs = zeros(10*8,1);
    nsolt = @(x) eznsolt.fcn_type1d222c55(nOrd,x,mus,isnodc);
else
    error('Not supported')
end

end