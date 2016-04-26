classdef Order2BuildingBlockTypeII < matlab.System
    %ORDER2BUILDINGBLOCKTYPEII Type-II building block with order 2
    %
    % SVN identifier:
    % $Id: Order2BuildingBlockTypeII.m 683 2015-05-29 08:22:13Z sho $
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

        function setupImpl(obj,~,~,~,~,~,~,~,p,~)
            obj.nHalfChannels = p;
            obj.nChannels = 2*p+1;
            obj.I = eye(p);
        end

        function output = stepImpl(obj,input,mtxHW,mtxHU,theta2,mtxW,mtxU,theta1,~,nshift)
            B = blkdiag(saivdr.dictionary.nsoltx.mexsrcs.fcn_build_butterfly_mtx(obj.nChannels,theta1),1);
            R = blkdiag(mtxW,mtxU,1);
            temp   = R*processQo_(obj,B,input, nshift);
            B = blkdiag(saivdr.dictionary.nsoltx.mexsrcs.fcn_build_butterfly_mtx(obj.nChannels,theta2),1);
            R = blkdiag(mtxHW,obj.I)*blkdiag(obj.I,mtxHU);
            output = R*processQe_(obj,B, temp, nshift);
        end

    end

    methods (Access = private)

        function value = processQo_(obj,B,x,nZ_)
            nch = obj.nChannels;
            hch = obj.nHalfChannels;
            x = B'*x;
            nLen = size(x,2);
            value = zeros([sum(nch) nLen+nZ_]);
            value(1:hch,1:nLen) = x(1:hch,:);
            value(hch+1:end-1,nZ_+1:end) = x(hch+1:end-1,:);
            value(end,1:nLen) = x(end,:);
            value = B*value;
        end

        function value = processQe_(obj,B,x,nZ_)
            nch = obj.nChannels;
            hch = obj.nHalfChannels;
            x = B'*x;
            nLen = size(x,2);
            value = zeros([sum(nch) nLen+nZ_]);
            value(1:hch,1:nLen) = x(1:hch,:);
            value(hch+1:end,nZ_+1:end) = x(hch+1:end,:);
            value = B*value;
        end
    end
end
