classdef PolyPhaseMatrix3d < handle
    %POLYPHASEMATRIX3D 3-D polyphase matrix
    %
    % SVN identifier:
    % $Id: PolyPhaseMatrix3d.m 683 2015-05-29 08:22:13Z sho $
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

    properties (Access = private, Constant = true)
       NUMBER_OF_DIMENSION = 3;
    end
    
    properties (GetAccess = public, SetAccess = private)
        Coefficients = [];
    end
    
    methods
        function obj = PolyPhaseMatrix3d(varargin)
            if nargin > 0
                input = varargin{1};
                if isa(input,'saivdr.dictionary.utility.PolyPhaseMatrix3d')
                    obj.Coefficients = input.Coefficients;
                else
                    obj.Coefficients = input;
                end
            end
        end
        
        function value = double(obj)
            value = double(obj.Coefficients);
        end
       
        function value = char(obj)
           coefSize = size(obj.Coefficients);
           value = ['PolyPhaseMatrix3d' 10];
           value = [value 9 'Number of rows   : ' num2str(coefSize(1)) 10];
           value = [value 9 'Number of colums : ' num2str(coefSize(2)) 10];
           value = [value 9 'Polyphase order  : ' ];
           for iDim=1:obj.NUMBER_OF_DIMENSION
               if length(coefSize) < 2+iDim
                   value = [value '0 '];
               else
                   value = [value num2str(coefSize(2+iDim)-1) ' '];
               end
           end
        end
        
        function disp(obj)
            disp([char(obj) 10]);
        end
        
        function value = subsref(obj,sub)
            % Implement a special subscripted assignment
            switch sub.type
                case '()'
                    r = sub.subs{1};
                    c = sub.subs{2};
                    value = permute(obj.Coefficients(r,c,:,:,:),[3 4 5 1 2]);
                case '.'
                    value = eval(sprintf('obj.%s',sub.subs));
                otherwise
                    error('Specify polyphase index for r,c,d as obj(r,c,d)')
            end
        end % subsref
        
        function value = plus(obj,another)
            import saivdr.dictionary.utility.PolyPhaseMatrix3d                                                           
            % Plus Implement obj1 + obj2 for PolyPhaseMatrix3d
            coef1 = double(obj);
            coef2 = double(another);
            if ndims(coef1) == ndims(coef2) && ...
                    all(size(coef1) == size(coef2)) || ...
                   (isscalar(coef1) || isscalar(coef2))
                  value = PolyPhaseMatrix3d(coef1+coef2);
            else
                [s1m1,s1m2,s1o1,s1o2,s1o3] = size(coef1);
                [s2m1,s2m2,s2o1,s2o2,s2o3] = size(coef2);
                s3 = max( [s1m1,s1m2,s1o1,s1o2,s1o3],...
                          [s2m1,s2m2,s2o1,s2o2,s2o3]);
                coef3 = zeros( s3 );
                n0 = size(coef1,3);
                n1 = size(coef1,4);
                n2 = size(coef1,5);
                coef3(:,:,1:n0,1:n1,1:n2) = coef1(:,:,1:n0,1:n1,1:n2);
                n0 = size(coef2,3);
                n1 = size(coef2,4);                
                n2 = size(coef2,5);                
                coef3(:,:,1:n0,1:n1,1:n2) = ...
                    coef3(:,:,1:n0,1:n1,1:n2) ...
                    + coef2(:,:,1:n0,1:n1,1:n2);
                value = PolyPhaseMatrix3d(coef3);
            end
        end % plus
        
        function value = minus(obj,another)
            value = plus(obj,-double(another));
        end % minus
        
        function value = mtimes(obj,another)
            import saivdr.dictionary.utility.PolyPhaseMatrix3d                                                           
            import saivdr.dictionary.utility.Direction
            coef1 = double(obj);
            coef2 = double(another);
            if size(coef1,2) ~= size(coef2,1)
                if ( isscalar(coef1) || isscalar(coef2) )
                    coef3 = coef1 * coef2;
                else
                    error('Inner dimensions must be the same as each other.');
                end
            else
                nDims = size(coef1,2);
                nRows = size(coef1,1);
                nCols = size(coef2,2);
                nCoefY = size(coef1,3)+size(coef2,3)-1;
                nCoefX = size(coef1,4)+size(coef2,4)-1;
                nCoefZ = size(coef1,5)+size(coef2,5)-1;
                pcoef1 = permute(coef1,[3 4 5 1 2]);
                pcoef2 = permute(coef2,[3 4 5 1 2]);
                pcoef3 = zeros(nCoefY,nCoefX,nCoefZ,nRows,nCols);
                for iCol = 1:nCols
                    for iRow = 1:nRows
                        array1 = pcoef1(:,:,:,iRow,1);
                        array2 = pcoef2(:,:,:,1,iCol);                        
                        array3 = convn(array1,array2);
                        for iDim = 2:nDims
                            array1 = pcoef1(:,:,:,iRow,iDim);
                            array2 = pcoef2(:,:,:,iDim,iCol);
                            array3 = array3 + convn(array1,array2);
                        end
                        pcoef3(:,:,:,iRow,iCol) = array3;
                    end
                end
                coef3 = ipermute(pcoef3,[3 4 5 1 2]);
            end
            value = PolyPhaseMatrix3d(coef3);
        end
        
        function value = ctranspose(obj)
            import saivdr.dictionary.utility.PolyPhaseMatrix3d            
            coefTmp = double(obj);    
            coefTmp = permute(coefTmp,[2 1 3 4 5]);
            coefTmp = flip(coefTmp,3);
            coefTmp = flip(coefTmp,4);
            coefTmp = flip(coefTmp,5);
            coefTmp = conj(coefTmp);
            value = PolyPhaseMatrix3d(coefTmp);
        end
        
        function value = transpose(obj)
            import saivdr.dictionary.utility.PolyPhaseMatrix3d                                                           
            coefTmp = double(obj);
            coefTmp = permute(coefTmp,[2 1 3 4 5]);
            coefTmp = flip(coefTmp,3);
            coefTmp = flip(coefTmp,4);
            coefTmp = flip(coefTmp,5);
            value = PolyPhaseMatrix3d(coefTmp);
        end
        
        function value = upsample(obj,ufactors,direction)
            import saivdr.dictionary.utility.Direction            
            import saivdr.dictionary.utility.PolyPhaseMatrix3d                                    
            value = obj;
            ucoef = obj.Coefficients;
            for iDirection = direction
                coefTmp = double(value);
                ufactor = ufactors(direction==iDirection);
                if (iDirection == Direction.VERTICAL) && ...
                        size(obj.Coefficients,3) ~= 1
                    uLength = size(coefTmp,3);
                    uLength = ufactor*(uLength - 1) + 1;
                    usize = size(coefTmp);
                    usize(3) = uLength;
                    ucoef = zeros(usize);
                    ucoef(:,:,1:ufactor:end,:,:) = coefTmp;
                elseif iDirection == Direction.HORIZONTAL && ...
                        size(obj.Coefficients,4) ~= 1
                    uLength = size(coefTmp,4);
                    uLength = ufactor*(uLength - 1) + 1;
                    usize = size(coefTmp);
                    usize(4) = uLength;
                    ucoef = zeros(usize);
                    ucoef(:,:,:,1:ufactor:end,:) = coefTmp;
                elseif iDirection == Direction.DEPTH && ...
                        size(obj.Coefficients,5) ~= 1
                    uLength = size(coefTmp,5);
                    uLength = ufactor*(uLength - 1) + 1;
                    usize = size(coefTmp);
                    usize(5) = uLength;
                    ucoef = zeros(usize);
                    ucoef(:,:,:,:,1:ufactor:end) = coefTmp;
                end
                value = PolyPhaseMatrix3d(ucoef);
            end
        end
    end
end
