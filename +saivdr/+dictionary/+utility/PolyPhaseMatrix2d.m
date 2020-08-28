classdef PolyPhaseMatrix2d < handle
    %POLYPHASEMATRIX2D 2-D polyphase matrix
    %
    % SVN identifier:
    % $Id: PolyPhaseMatrix2d.m 683 2015-05-29 08:22:13Z sho $
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
    
    properties (GetAccess = public, SetAccess = private)
        Coefficients = [];
    end
    
    methods
        function obj = PolyPhaseMatrix2d(input)
            if nargin == 1
                if isa(input,'saivdr.dictionary.utility.PolyPhaseMatrix2d')
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
            nRowsPhs = size(obj.Coefficients,1);
            nColsPhs = size(obj.Coefficients,2);
            value = ['[' 10]; % 10 -> \n
            if all(obj.Coefficients(:) == 0)
                value = '0';
            else
                for iRowPhs = 1:nRowsPhs
                    strrow = 9; % 9 -> \t
                    for iColPhs = 1:nColsPhs
                        coefMatrix = permute(...
                            obj.Coefficients(iRowPhs,iColPhs,:,:),[3 4 1 2]);
                        nOrdsY = size(coefMatrix,1) - 1;
                        nOrdsX = size(coefMatrix,2) - 1;
                        strelm = '0';
                        for iOrdX = 0:nOrdsX
                            for iOrdY = 0:nOrdsY
                                elm = coefMatrix(iOrdY+1,iOrdX+1);
                                if elm ~= 0
                                    if strelm == '0'
                                        strelm = [];
                                    end
                                    if ~isempty(strelm)
                                        if elm > 0
                                            strelm = [strelm ' + ' ];
                                        else
                                            strelm = [strelm ' - ' ];
                                            elm = -elm;
                                        end
                                    end
                                    if elm ~= 1 || (iOrdX == 0 && iOrdY == 0)
                                        strelm = [strelm num2str(elm)];
                                        if iOrdX > 0 || iOrdY > 0
                                            strelm = [strelm '*'];
                                        end
                                    end
                                    if iOrdY >=1
                                        strelm = [strelm 'y^(-' int2str(iOrdY) ')'];
                                        if iOrdX >=1
                                            strelm = [strelm '*'];
                                        end
                                    end
                                    if iOrdX >=1
                                        strelm = [strelm 'x^(-' int2str(iOrdX) ')'];
                                    end
                                end % for strelm ~= 0
                            end % for iOrdY
                        end % for iOrdX
                        strrow = [strrow strelm];
                        if iColPhs ~= nColsPhs
                            strrow = [strrow ',' 9]; % 9 -> \t
                        end
                    end % for iColPhs
                    if iRowPhs == nRowsPhs
                        value = [value strrow 10 ']']; % 10 -> \n
                    else
                        value = [value strrow ';' 10]; % 10 -> \n
                    end
                end % for iRowPhs
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
                    value = permute(obj.Coefficients(r,c,:,:),[3 4 1 2]);
                case '.'       
                    value = eval(sprintf('obj.%s',sub.subs));                    
                otherwise
                    error('Specify polyphase index for r,c as obj(r,c)')
            end
        end % subsref
        
        function value = plus(obj,another)
            import saivdr.dictionary.utility.PolyPhaseMatrix2d                                                           
            % Plus Implement obj1 + obj2 for PolyPhaseMatrix2d
            coef1 = double(obj);
            coef2 = double(another);
            if (ndims(coef1) == ndims(coef2) && ...
                    all(size(coef1) == size(coef2))) || ...
                    (isscalar(coef1) || isscalar(coef2))
                value = PolyPhaseMatrix2d(coef1+coef2);
            else
                [s1m1,s1m2,s1o1,s1o2] = size(coef1);
                [s2m1,s2m2,s2o1,s2o2] = size(coef2);
                s3 = max( [s1m1,s1m2,s1o1,s1o2],[s2m1,s2m2,s2o1,s2o2]);
                coef3 = zeros( s3 );
                n0 = size(coef1,3);
                n1 = size(coef1,4);
                coef3(:,:,1:n0,1:n1) = coef1(:,:,1:n0,1:n1);
                n0 = size(coef2,3);
                n1 = size(coef2,4);                
                coef3(:,:,1:n0,1:n1) = ...
                    coef3(:,:,1:n0,1:n1) + coef2(:,:,1:n0,1:n1);
                value = PolyPhaseMatrix2d(coef3);
            end
        end % plus
        
        function value = minus(obj,another)
            value = plus(obj,-double(another));
        end % minus
        
        function value = mtimes(obj,another)
            import saivdr.dictionary.utility.PolyPhaseMatrix2d                                                           
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
                pcoef1 = permute(coef1,[3 4 1 2]);
                pcoef2 = permute(coef2,[3 4 1 2]);
                pcoef3 = zeros(nCoefY,nCoefX,nRows,nCols);
                for iCol = 1:nCols
                    for iRow = 1:nRows
                        array1 = pcoef1(:,:,iRow,1);
                        array2 = pcoef2(:,:,1,iCol);
                        array3 = conv2(array1,array2);                        
                        for iDim = 2:nDims
                            array1 = pcoef1(:,:,iRow,iDim);
                            array2 = pcoef2(:,:,iDim,iCol);                            
                            array3 = array3 + conv2(array1,array2);
                        end
                        pcoef3(:,:,iRow,iCol) = array3;
                    end
                end
                coef3 = ipermute(pcoef3,[3 4 1 2]);
            end
            value = PolyPhaseMatrix2d(coef3);
        end
        
        function value = ctranspose(obj)
            import saivdr.dictionary.utility.PolyPhaseMatrix2d            
            coefTmp = double(obj);
            coefTmp = permute(coefTmp,[2 1 3 4]);
            coefTmp = flip(coefTmp,3);
            coefTmp = flip(coefTmp,4);
            coefTmp = conj(coefTmp);
            value = PolyPhaseMatrix2d(coefTmp);
        end
        
        function value = transpose(obj)
            import saivdr.dictionary.utility.PolyPhaseMatrix2d                                                           
            coefTmp = double(obj);
            coefTmp = permute(coefTmp,[2 1 3 4]);
            coefTmp = flip(coefTmp,3);
            coefTmp = flip(coefTmp,4);
            value = PolyPhaseMatrix2d(coefTmp);
        end
        
        function value = upsample(obj,ufactors,direction)
            import saivdr.dictionary.utility.Direction            
            import saivdr.dictionary.utility.PolyPhaseMatrix2d                                    
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
                    ucoef(:,:,1:ufactor:end,:) = coefTmp;
                elseif iDirection == Direction.HORIZONTAL && ...
                        size(obj.Coefficients,4) ~= 1
                    uLength = size(coefTmp,4);
                    uLength = ufactor*(uLength - 1) + 1;
                    usize = size(coefTmp);
                    usize(4) = uLength;
                    ucoef = zeros(usize);
                    ucoef(:,:,:,1:ufactor:end) = coefTmp;
                end
                value = PolyPhaseMatrix2d(ucoef);
            end
        end
    end
end
