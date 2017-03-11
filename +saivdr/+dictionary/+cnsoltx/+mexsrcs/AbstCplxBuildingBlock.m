classdef AbstCplxBuildingBlock < matlab.System  %#codegen
    %ABSTCPLXBUILDINGBLOCK  building block with
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
        I
        bmg
    end
    
    methods (Access = protected)
        
        function value = processQ_(obj,angles,x,nZ_)
            hChs = obj.nHalfChannels;
            nLen = size(x,2);
            B = butterflyMtx_(obj,angles);
            x = B'*x;
            value = complex(zeros([2*hChs nLen+nZ_]));
            value(1:hChs,1:nLen) = x(1:hChs,:);
            value(hChs+1:end,nZ_+1:end) = x(hChs+1:end,:);
            value = B*value/2;
        end

    end
    
    methods (Access = private)
        function hB = butterflyMtx_(obj,angles)
            hch = obj.nHalfChannels;
            hB = complex(zeros(2*hch));
            
            [ Cs, Ss] = step(obj.bmg,angles);
            for p = 1:floor(hch/2)
                range = 2*p-1:2*p;
                C = Cs(:,:,p);
                S = Ss(:,:,p);
                hB(range,range) = C;
                hB(range,range+hch) = conj(C);
                hB(range+hch,range) = S;
                hB(range+hch,range+hch) = conj(S);
            end
            if mod(hch,2) ~= 0
                hB(hch,hch) =  1;
                hB(hch,end) =  1;
                hB(end,hch) =  1i;
                hB(end,end) = -1i;
            end
        end
    end
    
%     methods (Static = true)
%         function hB = butterflyMtx(hch, angles)
%             hC = complex(eye(hch));
%             hS = complex(eye(hch));
%             for p = 1:floor(hch/2)
%                 tp = angles(p)/2 + pi/4;
%                 
%                 hC(2*p-1:2*p, 2*p-1:2*p) = [-1i*cos(tp), -1i*sin(tp);
%                     cos(tp) , -sin(tp)]; %c^
%                 hS(2*p-1:2*p, 2*p-1:2*p) = [ -1i*sin(tp), -1i*cos(tp);
%                     sin(tp) , -cos(tp)]; %s^
%             end
%             
%             hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
%         end
%         
%     end
end
