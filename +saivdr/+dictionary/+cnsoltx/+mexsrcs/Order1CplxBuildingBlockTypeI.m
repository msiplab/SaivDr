classdef Order1CplxBuildingBlockTypeI < saivdr.dictionary.nsoltx.mexsrcs.AbstBuildingBlock  %#codegen
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

    methods (Access = protected)
        function setupImpl(obj,~,~,~,~,p,~)
            obj.nHalfChannels = p;
            obj.I             = eye(p);
        end

	    function output = stepImpl(obj,input,mtxW,mtxU,angles,~,nshift)
            output = processQ_(obj,angles,input,nshift);
            
            hCh = obj.nHalfChannels;
            output(1:hCh,:) = mtxW*output(1:hCh,:);
            output(hCh+1:end,:) = mtxU*output(hCh+1:end,:);
        end
    end

end
