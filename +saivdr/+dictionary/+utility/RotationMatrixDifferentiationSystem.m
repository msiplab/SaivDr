classdef RotationMatrixDifferentiationSystem < matlab.System %#codegen
    %RotationMatrixDifferentiationSYSTEM Orthonormal matrix generator
    %
    % SVN identifier:
    % $Id: RotationMatrixDifferentiationSystem.m 683 2015-05-29 08:22:13Z sho $
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
        function obj = RotationMatrixDifferentiationSystem(varargin)
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
        
        function setupImpl(obj,angles,~,~)
            obj.NumberOfDimensions = (1+sqrt(1+8*length(angles)))/2;
        end
        
        function validateInputsImpl(~,angles,mus,idx)
            if ~isempty(mus) && any(abs(mus(:))~=1)
                error('All entries of mus must be 1 or -1.');
            end
            nDim_ = (1+sqrt(1+8*length(angles)))/2;
            nAngs_ = nDim_*(nDim_-1)/2;
            if ~isscalar(idx) || idx < 1 || idx > nAngs_
                error('Idx must be a natual number less than or equal to %d.',nAngs_);
            end
        end
        
        function matrix = stepImpl(obj,angles,mus,idx)
            
            nDim_ = obj.NumberOfDimensions;
            %
            if isempty(angles)
                angles = zeros(nDim_*(nDim_-1)/2,1);
            end
            %
            matrix = eye(nDim_);
            iAng = 1;
            for iTop=1:nDim_-1
                vt = matrix(iTop,:);
                for iBtm=iTop+1:nDim_
                    if iAng == idx
                        angle = angles(iAng)+pi/2;
                        c = cos(angle); %
                        s = sin(angle); %
                        vb = matrix(iBtm,:);
                        %
                        u  = s*(vt + vb);
                        vt = (c + s)*vt;
                        vb = (c - s)*vb;
                        vt = vt - u;
                        %
                        matrix = zeros(size(matrix));
                        matrix(iBtm,:) = vb + u;
                    else
                        angle = angles(iAng);
                        c = cos(angle); %
                        s = sin(angle); %
                        vb = matrix(iBtm,:);
                        %
                        u  = s*(vt + vb);
                        vt = (c + s)*vt;
                        vb = (c - s)*vb;
                        vt = vt - u;
                        matrix(iBtm,:) = vb + u;
                    end
                    iAng = iAng + 1;
                end
                matrix(iTop,:) = vt;
            end
            matrix = diag(mus)*matrix;
            
        end
        
        function N = getNumInputsImpl(~)
            N = 3;
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end   
    end
end
