function [ grad ] = fcn_GradEvalSteps2d( ...  %#codegen
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, nch, ord, fpe, isnodc)
%FCN_GRADEVALSTEPS2D

%%
persistent ges
if isempty(ges) 
    ges = saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d(...
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

