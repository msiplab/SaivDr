function analysislgraph = fcn_cpparamssyn2ana(analysislgraph,synthesislgraph)
%FCN_CPPARAMSSYN2ANA
%
% Setting up the analysis dictionary (adjoint operator) by copying
% synthesis dictionary parameters to the analyisis dictionary
%
% Requirements: MATLAB R2020a
%
% Copyright (c) 2020-2021, Shogo MURAMATSU
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
expanalyzer = '^Lv\d+_Cmp\d+_V(\w\d|0)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
    alayer = analysislgraph.Layers(iLayer);
    alayerName = alayer.Name;
    if ~isempty(regexp(alayerName,expanalyzer,'once'))
        slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
        alayer.Angles = slayer.Angles;
        alayer.Mus = slayer.Mus;
        if isa(alayer,'saivdr.dcnn.nsoltInitialRotation2dLayer')
            alayer.NoDcLeakage = slayer.NoDcLeakage;
        end
        analysislgraph = analysislgraph.replaceLayer(alayerName,alayer);
        disp("Copy angles from " + slayer.Name + " to " + alayerName)
    end
end
end