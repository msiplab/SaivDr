classdef OrthonormalMatrixFactorizationSystem < matlab.System %#codegen
    %ORTHONORMALMATRIXFACTORIZATIONSYSTEM Orthonormal matrix factorizer
    %
    % SVN identifier:
    % $Id: OrthonormalMatrixFactorizationSystem.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (Nontunable)
        OrderOfProduction = 'Descending';
    end
    
    properties (Access = private, Transient)
        OrderOfProductionSet = ...
            matlab.system.StringSet({...
            'Ascending',...
            'Descending'});
    end
    
    methods
        function obj = OrthonormalMatrixFactorizationSystem(varargin)
            % Support name-value pair arguments
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
        
        function setupImpl(obj,matrix)
            obj.NumberOfDimensions = size(matrix.',2);
        end
        
        function [angles,mus] = stepImpl(obj,matrix)
            if strcmp(obj.OrderOfProduction,'Descending')
                T = matrix.';
            elseif strcmp(obj.OrderOfProduction,'Ascending')
                T = matrix;
            else
                % TODO: ????
            end
            
            iAng = 1;
            nDim = obj.NumberOfDimensions;
            angles = zeros(nDim*(nDim-1)/2,1);
            for iCol = 1:nDim-1
                v = T(iCol:end,iCol);
                for idx = 2:length(v)
                    x = [ v(1) ; v(idx) ];
                    [G,y] = planerot(x);
                    angles(iAng) = atan2(G(2,1),G(1,1));
                    v(1) = y(1);
                    v(idx) = y(2);
                    R = eye(nDim);
                    iRow = iCol+idx-1;
                    R(iCol,iCol) = G(1,1);
                    R(iRow,iCol) = G(2,1);
                    R(iCol,iRow) = G(1,2);
                    R(iRow,iRow) = G(2,2);
                    T = R*T;
                    iAng = iAng + 1;
                end
            end
            mus = round(diag(T));
        end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 2;
        end
    end
    
end
