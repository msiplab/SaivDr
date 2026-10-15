function s = genista(c)
%GENISTA ISTA-based restoration as examples/rgbdemos/quickstart_rgb_*.m
%
%   A synthetic image is degraded and restored with MaxIter iterations by
%   IstaImRestoration2d, with the generalfb analyzer and synthesizer of a
%   pre-designed nsoltx filter bank.
[X,Y] = meshgrid(linspace(0,1,c.size(2)),linspace(0,1,c.size(1)));
orgImg = zeros(c.size);
for iCmp = 1:c.size(3)
    orgImg(:,:,iCmp) = 0.5+0.3*sin(2*pi*(3+iCmp)*X.*(1+Y)).*(X>0.3)+0.2*(Y>0.6);
end
orgImg = min(max(orgImg + 0.05*randn(c.size),0),1);
switch c.degradation
    case 'db'
        linproc = saivdr.degradation.linearprocess.BlurSystem(...
            'BlurType','Gaussian','SigmaOfGaussianKernel',2);
    case 'sr'
        linproc = saivdr.degradation.linearprocess.DecimationSystem(...
            'DataType','Image','TolOfPowerMethod',1e-6,...
            'VerticalDecimationFactor',2,'HorizontalDecimationFactor',2,...
            'BlurType','Gaussian','SigmaOfGaussianKernel',2);
    case 'ip'
        linproc = saivdr.degradation.linearprocess.PixelLossSystem(...
            'LossType','Random','Density',0.4,'Seed',0);
end
noiseVar = (c.nsigma/255)^2;
awgn = saivdr.degradation.noiseprocess.AdditiveWhiteGaussianNoiseSystem(...
    'Mean',0,'Variance',noiseVar);
dgrd = saivdr.degradation.DegradationSystem('LinearProcess',linproc,'NoiseProcess',awgn);
obsImg = step(dgrd,orgImg);
d = load(c.dictionary,'lppufbs');
lppufb = saivdr.dictionary.utility.fcn_upgrade(d.lppufbs{end});
file = [tempname '.mat'];
saivdr_export_nsoltx(lppufb,file,'NumberOfLevels',c.nLevels);
s = load(file);
delete(file);
nDec = get(lppufb,'DecimationFactor');
release(lppufb); lppufb.OutputMode = 'AnalysisFilters';
analysisFilters = step(lppufb,[],[]);
release(lppufb); lppufb.OutputMode = 'SynthesisFilters';
synthesisFilters = step(lppufb,[],[]);
analyzer = saivdr.dictionary.generalfb.Analysis2dSystem('DecimationFactor',nDec,...
    'AnalysisFilters',analysisFilters,'NumberOfLevels',c.nLevels,'FilterDomain','Frequency');
analyzer.UseGpu = false;
synthesizer = saivdr.dictionary.generalfb.Synthesis2dSystem('DecimationFactor',nDec,...
    'SynthesisFilters',synthesisFilters,'FilterDomain','Frequency');
setFrameBound(synthesizer,1);
synthesizer.UseGpu = false;
ista = saivdr.restoration.ista.IstaImRestoration2d('Synthesizer',synthesizer,...
    'AdjOfSynthesizer',analyzer,'LinearProcess',linproc,'Lambda',c.lambda,...
    'MaxIter',c.maxIter);
resImg = step(ista,obsImg);
s.meta_degradation = c.degradation;
s.lambda = c.lambda;
s.max_iter = c.maxIter;
s.original = orgImg;
s.original_size = refgen.padsize(orgImg,4);
s.observed = obsImg;
s.observed_size = refgen.padsize(obsImg,4);
s.restored = resImg;
s.restored_size = refgen.padsize(resImg,4);
s.lambda_max = get(linproc,'LambdaMax');
if strcmp(c.degradation,'ip')
    s.mask = double(step(clone(linproc),ones(c.size(1:2))) ~= 0);
end
end
