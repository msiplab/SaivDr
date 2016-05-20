classdef Order2BuildingBlockTypeII < saivdr.dictionary.nsoltx.mexsrcs.AbstBuildingBlock
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

    methods (Access = protected)

        function setupImpl(obj,~,~,~,~,~,~,~,p,~)
            obj.nHalfChannels = p;
            obj.I = eye(p);
        end

        function output = stepImpl(obj,input,mtxHW,mtxHU,angles2,mtxW,mtxU,angles1,~,nshift)
            R = blkdiag(mtxW,mtxU);
            temp1 = R*processQ_(obj,angles1,input(1:end-1,:), nshift);
            
            temp2u = processQ_(obj, angles2, temp1, nshift);
            temp2b = horzcat(zeros(1,nshift),input(end,:), zeros(1,nshift));
            R = blkdiag(mtxHW,obj.I)*blkdiag(obj.I,mtxHU);
            output = R*[temp2u ; temp2b];
        end

    end

end
