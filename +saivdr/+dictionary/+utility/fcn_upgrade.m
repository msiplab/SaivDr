function [ outputFb, isConversion ] = fcn_upgrade( inputFb )
%NSOLTX_OVSDLPPUFB2D Converter of Old Package Systems
%
% SVN identifier:
% $Id: fcn_upgrade.m 683 2015-05-29 08:22:13Z sho $
%
% Requirements: MATLAB R2015b
%
% Copyright (c) 2014-2015, Shogo MURAMATSU
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
if strfind(class(inputFb),'saivdr.dictionary.nsolt.')
    isConversion = true;
    isOvsdLpPuFb2d = true;
    if strfind(class(inputFb),'Vm1')
        vm = 1;
    else
        vm = 0;
    end
elseif strfind(class(inputFb),'saivdr.dictionary.nsgenlot.')
    isConversion = true;
    isOvsdLpPuFb2d = false;
    if strfind(class(inputFb),'Tvm')
        vm = 2;
        angleTvm = get(inputFb,'TvmAngleInDegree');
    elseif strfind(class(inputFb),'Vm2')
        vm = 2;
        angleTvm = [];
    elseif strfind(class(inputFb),'Vm1')
        vm = 1;
    else
        vm = 0;
    end
else
    isConversion = false;
end
%
if isConversion
    warning('Temporary upgrading ...')
    dec = get(inputFb,'DecimationFactor');
    ord = get(inputFb,'PolyPhaseOrder');
    omd = get(inputFb,'OutputMode');
    ang = get(inputFb,'Angles');
    mus = get(inputFb,'Mus');
    if isOvsdLpPuFb2d
        import saivdr.dictionary.nsoltx.NsoltFactory
        nch = get(inputFb,'NumberOfChannels');
        outputFb = NsoltFactory.createOvsdLpPuFb2dSystem(...
            'DecimationFactor',dec,...
            'NumberOfChannels',nch,...
            'PolyPhaseOrder',ord,...
            'NumberOfVanishingMoments',vm,....
            'OutputMode',omd);
        set(outputFb,'Angles', ang,'Mus', mus);
    else
        import saivdr.dictionary.nsgenlotx.NsGenLotFactory
        outputFb = NsGenLotFactory.createLpPuFb2dSystem(...
            'DecimationFactor',dec,...
            'PolyPhaseOrder',ord,...
            'NumberOfVanishingMoments',vm,....
            'TvmAngleInDegree',angleTvm,...
            'OutputMode',omd);
        set(outputFb,'Angles', ang,'Mus', mus);
    end
    step(outputFb,[],[]);
else
    outputFb = inputFb;
end
end

