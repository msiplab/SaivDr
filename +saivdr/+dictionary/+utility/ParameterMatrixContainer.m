classdef ParameterMatrixContainer < matlab.System %#codegen
    %PARAMETERMATRIXCONTAINER Parameter matrix set
    %
    % Requirements: MATLAB R2015b
    %
    % Copyright (c) 2014-2016, Shogo MURAMATSU
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
    
    properties (Nontunable)
        MatrixSizeTable = [ 2 2 ];
    end
    
    properties (Hidden)
        Coefficients
    end
    
    properties (GetAccess = public, SetAccess = protected, PositiveInteger)
        NumberOfParameterMatrices
    end

    properties (Access = protected)
        indexSizeTable
    end

    methods
             
        function obj = ParameterMatrixContainer(varargin)
            setProperties(obj,nargin,varargin{:})
            mtxSzTab_ = obj.MatrixSizeTable;
            obj.Coefficients = zeros(sum(prod(mtxSzTab_,2)),1);
            nRows = size(mtxSzTab_,1);
            obj.indexSizeTable = zeros(nRows,3);
            cidx = 1;
            for iRow = uint32(1):nRows
                obj.indexSizeTable(iRow,:) = ...
                    [ cidx mtxSzTab_(iRow,:)];
                cidx = cidx + prod(mtxSzTab_(iRow,:));
            end
            obj.NumberOfParameterMatrices = nRows;    
        end
        
    end

    methods (Access = protected)

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.indexSizeTable = obj.indexSizeTable;
            s.NumberOfParameterMatrices = obj.NumberOfParameterMatrices;
        end
        
        function loadObjectImpl(obj, s, wasLocked)
            obj.indexSizeTable = s.indexSizeTable;
            obj.NumberOfParameterMatrices = s.NumberOfParameterMatrices;
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end
        
        function setupImpl(~,~,~)
        end
        
        function output = stepImpl(obj,input,index)
            idxSzTab_ = obj.indexSizeTable;
            startIdx  = idxSzTab_(index,1);
            dimension = idxSzTab_(index,2:3);
            nElements = prod(dimension); 
            endIdx = startIdx + nElements - 1;
            if ~isempty(input)
                if size(input) == dimension
                    obj.Coefficients(startIdx:endIdx) = ...
                        input(:).';
                    output = [];
                else
                    error('Invalid size of input array.');
                end
            else
                output = reshape(...
                    obj.Coefficients(startIdx:endIdx),...
                    dimension);
            end
        end        
        
        function N = getNumInputsImpl(~)
            N = 2;
        end
        
        function N = getNumOutputsImpl(~)
            N = 1;
        end
          
    end

end
