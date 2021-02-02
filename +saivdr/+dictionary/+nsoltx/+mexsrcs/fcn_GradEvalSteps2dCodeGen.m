function [ grad ] = fcn_GradEvalSteps2dCodeGen( ...  %#codegen
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, nch, ord, fpe, isnodc)
%FCN_GRADEVALSTEPS2D

%%
persistent ges
if isempty(ges) 
    ges = saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps2d();
end

%%
set(ges,'NumberOfSymmetricChannels',nch(1));
set(ges,'NumberOfAntisymmetricChannels',nch(2));
set(ges,'PolyPhaseOrder',ord);
set(ges,'IsPeriodicExt',fpe);

%%
grad = step(ges, ...
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, isnodc );

