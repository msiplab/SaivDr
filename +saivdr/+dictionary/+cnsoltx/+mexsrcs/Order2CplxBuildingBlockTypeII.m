classdef Order2CplxBuildingBlockTypeII < saivdr.dictionary.cnsoltx.mexsrcs.AbstBuildingBlock
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

        function output = stepImpl(obj,input,mtxWE,mtxUE,angB1,mtxWO,mtxHO,angB2,~,nshift)
            hCh = obj.nHalfChannels;
            tmp1 = processQ_(obj,angB1,input(1:end-1,:), nshift);
            tmp1(1:hCh,:) = mtxWE * tmp1(1:hCh,:);
            tmp1(hCh+1:end,:) = mtxUE * tmp1(hCh+1:end,:);
            
            tmp2u = processQ_(obj, angB2, tmp1, nshift);
            tmp2l = horzcat(zeros(1,nshift),input(end,:), zeros(1,nshift));
            output = [tmp2u ; tmp2l];
            output(hCh+1:end,:) = mtxHO*output(hCh+1:end,:);
            output(1:hCh+1,:) = mtxWO*output(1:hCh+1,:);
        end

    end

end
