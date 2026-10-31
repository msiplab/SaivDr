function saivdr_export_params(net,file)
%SAIVDR_EXPORT_PARAMS Export NSOLT parameters for saivdr.dcnn (PyTorch)
%
%   saivdr_export_params(net,file) saves the Angles, Mus and NoDcLeakage of
%   all the rotation layers of the dlnetwork NET to the MAT-file FILE, which
%   saivdr.dcnn.matlab_compat.load_params reads in Python.
%
% Copyright (c) 2026, Shogo MURAMATSU
s = refgen.collectparams(net,struct());
save(file,'-struct','s','-v7');
end
