function analysisnet = fcn_cpparamssyn2ana(synthesisnet,analysisnet)
%FCN_CPPARAMSSYN2ANA
%
% Setting up the analysis dictionary (adjoint operator) by copying
% synthesis dictionary parameters to the analyisis dictionary
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
import saivdr.dcnn.*
%
synthesisLearnables = synthesisnet.Learnables;
nLearnables = height(synthesisLearnables);
for iLearnable = 1:nLearnables
    layer = synthesisLearnables.Layer(iLearnable);
    value = synthesisLearnables.Value(iLearnable);
    idx = (analysisnet.Learnables.Layer == strrep(layer,'~',''));
    analysisnet.Learnables.Value(idx) = value;
end
end