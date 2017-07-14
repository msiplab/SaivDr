%MYBUILD Script for building mex codes
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2017, Shogo MURAMATSU
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
    %
    import saivdr.dictionary.colpprfb.mexsrcs.*
    fcn_build_catomext1d;
    fcn_build_catomcnc1d;
    %
    import saivdr.dictionary.cnsoltx.mexsrcs.*
    fcn_build_catomext2d;
    fcn_build_catomcnc2d;
    fcn_build_catomext3d;
    fcn_build_catomcnc3d;
    fcn_build_cbb_type1;
    fcn_build_cbb_type2;
    %
    import saivdr.dictionary.colpprfb.mexsrcs.*
    fcn_build_catomext1d;
    fcn_build_catomcnc1d;
    %
    import saivdr.dictionary.cnsoltx.mexsrcs.*
    fcn_build_catomext2d;
    fcn_build_catomcnc2d;
    fcn_build_catomext3d;
    fcn_build_catomcnc3d;
    fcn_build_cbb_type1;
    fcn_build_cbb_type2;
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
    import saivdr.dictionary.colpprfb.mexsrcs.*
    fcn_build_catomext1d;
    fcn_build_catomcnc1d;
    %
    import saivdr.dictionary.cnsoltx.mexsrcs.*
    fcn_build_catomext2d;
    fcn_build_catomcnc2d;
    fcn_build_catomext3d;
    fcn_build_catomcnc3d;
    fcn_build_cbb_type1;
    fcn_build_cbb_type2;
    %
end
