classdef AbstBuildingBlock < matlab.System  %#codegen
    %ABSTBUILDINGBLOCK  building block with
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
        function setupImpl(obj,~,~,p,~)
            obj.nHalfChannels = p;
            obj.nChannels     = 2*p;
            obj.I             = eye(p);
        end

        function value = processQ_(obj,B,x,nZ_)
            hChs = obj.nHalfChannels;
            nChs = obj.nChannels;
            nLen = size(x,2);
            x = B'*x;
            value = zeros([nChs nLen+nZ_]);
            value(1:hChs,1:nLen) = x(1:hChs,:);
            value(hChs+1:end,nZ_+1:end) = x(hChs+1:end,:);
            value = B*value;
        end
        
        function hB = mtxB_(obj,P,theta)
            qtrP = floor(P/4);
	
            hC = [];
            hS = [];
            for p = 1:qrtP
                tp = theta(p);
        
                hC = blkdiag(hC, _buildMtxHc(tp));
                hS = blkdiag(hS, _buildMtxHs(tp));
            end
	
            if odd(qtrP)
                hC = blkdiag(hC, 1);
                hS = blkdiag(hS, 1);
            end
            hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
        end

        function mtxHc = buildMtxHc_(t)
            mtxHc =     [-1i*cos(t), -1i*sin(t);
                        cos(t) , -sin(t)];
        end

        function mtxHs = buildMtxHs_(t)
            mtxHs =     [ -1i*sin(t), -1i*cos(t);
                        sin(t) , -cos(t)];
        end

    end
end
