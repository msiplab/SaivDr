function s = gensparserep(c)
%GENSPARSEREP Sparse approximation by IterativeHardThresholding
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
synthesizer = saivdr.dictionary.generalfb.Synthesis2dSystem('DecimationFactor',nDec,...
    'SynthesisFilters',synthesisFilters,'FilterDomain','Frequency');
[X,Y] = meshgrid(linspace(0,1,c.size(2)),linspace(0,1,c.size(1)));
src = 0.5+0.3*sin(2*pi*4*X.*(1+Y)).*(X>0.3)+0.2*(Y>0.6)+0.05*randn(c.size);
iht = saivdr.sparserep.IterativeHardThresholding('Dictionary',{synthesizer,analyzer},...
    'NumberOfSparseCoefficients',c.nCoefs,'MaxIter',c.maxIter);
[result,coefvec] = step(iht,src);
s.source = src;
s.result = result;
s.coefs = coefvec(:);
s.n_coefs = c.nCoefs;
s.max_iter = c.maxIter;
end
