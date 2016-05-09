classdef Order1BuildingBlockTypeI < matlab.System  %#codegen
    %ORDER1BUILDINGBLOCKTYPEI  Type-I building block with order 1
    %
    % SVN identifier:
    % $Id: Order1BuildingBlockTypeI.m 683 2015-05-29 08:22:13Z sho $
    %
    % Requirements: MATLAB R2013b
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
    % LinedIn: http://www.linkedin.com/pub/shogo-muramatsu/4b/b08/627
    %

    properties (Access=protected,Nontunable)
        nHalfChannels
        nChannels
        I
    end

    methods (Access = protected)
        function setupImpl(obj,~,~,~,~,p,~)
            obj.nHalfChannels = p;
            obj.nChannels     = 2*p;
            obj.I             = eye(p);
        end

	    function output = stepImpl(obj,input,mtxW,mtxU,angles,~,nshift)
            R = blkdiag(mtxW,mtxU);
            hB = saivdr.dictionary.nsoltx.mexsrcs.fcn_build_butterfly_mtx(obj.nChannels, angles);
            output = R*processQ_(obj,hB,input,nshift);
        end
    end

    methods (Access = private)

        function value = processQ_(obj,B,x,nZ_)
            hChs = obj.nHalfChannels;
            nChs = obj.nChannels;
            nLen = size(x,2);
            x = B'*x;
            value = complex(zeros([nChs nLen+nZ_]));
            value(1:hChs,1:nLen) = x(1:hChs,:);
            value(hChs+1:end,nZ_+1:end) = x(hChs+1:end,:);
            value = B*value;
        end
    end
end
