%MYBUILD Script for building mex codes
%
% Requirements: MATLAB R2021a
%
% Copyright (c) 2017-2021, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%

%% Set path
setpath

%% Build mex codes
if license('checkout','matlab_coder')
    import saivdr.dcnn.mexsrcs.*
    datatypes = { 'single', 'double' };
    for idatatype = 1:length(datatypes)
        for useGpuArray = 0:1
            fcn_build_orthmtxgen(datatypes{idatatype},useGpuArray);
            fcn_build_orthmtxgen_diff(datatypes{idatatype},useGpuArray);
        end
    end
    %
    import saivdr.dictionary.olpprfb.mexsrcs.*
    fcn_build_atomext1d;
    fcn_build_atomcnc1d;
    %
    import saivdr.dictionary.nsoltx.mexsrcs.*
    fcn_build_atomext2d;
    fcn_build_atomcnc2d;
    fcn_build_atomext3d;
    fcn_build_atomcnc3d;    
    fcn_build_bb_type1;
    fcn_build_bb_type2;
    fcn_build_gradevalsteps2d;
    fcn_build_gradevalsteps3d;
    %
end
