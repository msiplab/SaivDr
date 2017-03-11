classdef ButterflyMatrixGenerationSystem < matlab.System
    %BUTTERFLYMATRIXGENERATIONSYSTEM このクラスの概要をここに記述
    %   詳細説明をここに記述
    
    properties (Nontunable)
        NumberOfSubmatrices
    end
    
    methods
        function obj = ButterflyMatrixGenerationSystem(varargin)
            setProperties(obj,nargin,varargin{:});
        end
    end
    
    methods (Access = protected)
%         function setupImpl(obj,angles)
%             %obj.NumberOfSubmatrices = length(angles);
%         end
        
        function [ Cs, Ss ] = stepImpl(obj,angles)
            n = obj.NumberOfSubmatrices;
            Cs = complex(zeros(2,2,n));
            Ss = complex(zeros(2,2,n));
            tp = angles/2+pi/4;
            for idx = 1:n
                Cs(:,:,idx) = subMatrixC_(obj,tp(idx));
                Ss(:,:,idx) = subMatrixS_(obj,tp(idx));
            end
        end
        
%         function validatePropertiesImpl(obj)
%             if mod(obj.NumberOfDimensions,2) ~= 0
%                 error('NumberOfDimensions must be even.');
%             end
%         end
        
        function N = getNumInputsImpl(~)
            N = 1;
        end
        
        function N = getNumOutputsImpl(~)
            N = 2;
        end
        
    end
    
    methods (Access = private)
        function value = subMatrixC_(~,angle)
            value = [
                -1i*cos(angle), -1i*sin(angle);
                    cos(angle),    -sin(angle)];
        end
        
        function value = subMatrixS_(~,angle)
            value = [
                   sin(angle),     cos(angle);
                1i*sin(angle), -1i*cos(angle)];
        end
        
    end
    
end

