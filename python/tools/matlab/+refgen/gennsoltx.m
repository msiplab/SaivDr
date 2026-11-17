function s = gennsoltx(c)
%GENNSOLTX Export an nsoltx filter bank and the generalfb analysis/synthesis
if strcmp(c.source,'random')
    lppufb = saivdr.dictionary.nsoltx.NsoltFactory.createOvsdLpPuFb2dSystem(...
        'DecimationFactor',[2 2],'NumberOfChannels',c.nChs,...
        'PolyPhaseOrder',c.ppo,'NumberOfVanishingMoments',c.nVm,...
        'OutputMode','ParameterMatrixSet');
    angles = get(lppufb,'Angles');
    set(lppufb,'Angles',pi*(2*rand(size(angles))-1));
    mus = get(lppufb,'Mus');
    mus = sign(randn(size(mus))); mus(mus==0) = 1;
    if c.nVm == 1
        mus(1,1) = 1;
    end
    set(lppufb,'Mus',mus);
else
    d = load(c.source,'lppufbs');
    lppufb = saivdr.dictionary.utility.fcn_upgrade(d.lppufbs{end});
end
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
x = rand(c.size);
[coefs,scales] = step(analyzer,x);
y = randn(size(coefs));
s.x = x;
s.coefs = coefs(:);
s.scales = scales;
s.y = y(:);
s.synthesized = step(synthesizer,y,scales);
s.meta_source = c.source;
end
