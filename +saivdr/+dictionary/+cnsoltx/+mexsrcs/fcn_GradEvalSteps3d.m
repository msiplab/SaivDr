function [ grad ] = fcn_GradEvalSteps3d( ...  %#codegen
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, nch, ord, fpe, isnodc)
%FCN_GRADEVALSTEPS3D

%%
persistent ges
if isempty(ges) 
    ges = saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d(...
        'NumberOfSymmetricChannels',nch(1),...
        'NumberOfAntisymmetricChannels',nch(2),...
        'PolyPhaseOrder',ord);
end

%%
set(ges,'IsPeriodicExt', fpe);

%%
grad = step(ges, ...
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, isnodc );

