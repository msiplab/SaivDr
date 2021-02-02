function [ grad ] = fcn_GradEvalSteps3dCodeGen( ...  %#codegen
    arrayCoefsB, arrayCoefsC, scale, pmCoefs, ...
    angs, mus, nch, ord, fpe, isnodc)
%FCN_GRADEVALSTEPS3D

%%
persistent ges
if isempty(ges) 
    ges = saivdr.dictionary.nsoltx.mexsrcs.GradEvalSteps3d();
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

