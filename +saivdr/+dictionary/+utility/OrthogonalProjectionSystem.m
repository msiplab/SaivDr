classdef OrthogonalProjectionSystem < matlab.System %#codegen
    %ORTHOGONALPROJECTINOSYSTEM Orthogonal projection
    %
    % SVN identifier:
    % $Id: OrthogonalProjectionSystem.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties(Access = protected,Nontunable)
        NumberOfDimensions
    end
    
    methods
        function obj = OrthogonalProjectionSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.NumberOfDimensions = obj.NumberOfDimensions;
        end
        
        function loadObjectImpl(obj,s,wasLocked)
            obj.NumberOfDimensions = s.NumberOfDimensions;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(obj,vector)
            obj.NumberOfDimensions = length(vector);
        end
        
        function [angles, mus] = stepImpl(obj,vector)

            P = eye(obj.NumberOfDimensions);
            angles = zeros(obj.NumberOfDimensions*(obj.NumberOfDimensions-1)/2,1);
            mus = ones(obj.NumberOfDimensions,1);
            iAng = 1;
            for idx = 2:length(vector)
                x = [ vector(1) ; vector(idx) ];
                [G,y] = planerot(x);
                angles(iAng) = atan2(G(2,1),G(1,1));
                vector(1) = y(1);
                vector(idx) = y(2);
                R = eye(obj.NumberOfDimensions);
                iRow = idx;
                R(1,1) = G(1,1);
                R(iRow,1) = G(2,1);
                R(1,iRow) = G(1,2);
                R(iRow,iRow) = G(2,2);
                P = R*P;
                iAng = iAng + 1;
            end
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 2;
        end        
    end

end
