classdef nsoltIntermediateRotation2dLayerBoundaryOperationTestCase < matlab.unittest.TestCase
    %NSOLTINTERMEDIATEROTATION2DLAYERBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of nsoltIntermediateRotation2dLayer.
    %
    %   Termination replaces the rotation of the boundary slice with -I and
    %   excludes it from the angle gradients, as in
    %   saivdr.dictionary.nsoltx.NsoltAtomExtender2d.
    %   The boundary is specified by Direction:
    %      'Horizontal' : first column, X(:,:,1,:)
    %      'Vertical'   : first row,    X(:,1,:,:)
    %
    %   コンポーネント別に入力(nComponents):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChsTotal x nRows x nCols x nSamples
    %
    % References:
    %   S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary
    %   Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks,"
    %   IEEE Trans. Image Process., vol. 21, no. 4, pp. 2314-2318, 2012.
    %
    %   K. Furuya, S. Hara, K. Seino and S. Muramatsu, "Boundary operation of
    %   2D non-separable oversampled lapped transforms," APSIPA Trans. Signal
    %   Inf. Process., vol. 5, no. 1, pp. 1-9, 2016.
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2026, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    properties (TestParameter)
        nchs = { [3 3], [4 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 4,'medium', 8, 'large', 16);
        ncols = struct('small', 4,'medium', 8, 'large', 16);
        mode = { 'Analysis', 'Synthesis' };
        bdir = { 'Horizontal', 'Vertical' };
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import saivdr.dcnn.*
            for direction = [ "Horizontal", "Vertical" ]
                layer = nsoltIntermediateRotation2dLayer(...
                    'NumberOfChannels',[3 3],...
                    'BoundaryOperation','Termination',...
                    'Direction',direction);
                fprintf("\n --- Check layer for 2-D images (Termination, %s) ---\n",...
                    direction);
                checkLayer(layer,[6 8 8],...
                    'ObservationDimension',4,...
                    'CheckCodegenCompatibility',true)
            end
        end
    end

    methods (Test)

        function testConstructorDefaultBoundaryOperation(testCase, nchs)
            % Default BoundaryOperation is 'Circular' (backward compatibility)

            % Expected values
            expctdBoundaryOperation = 'Circular';

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs);

            % Actual values
            actualBoundaryOperation = layer.BoundaryOperation;

            % Evaluation
            testCase.verifyEqual(actualBoundaryOperation,expctdBoundaryOperation);
        end

        function testConstructorWithBoundaryOperation(testCase, nchs, bdir)
            % BoundaryOperation and Direction are set by the constructor

            % Expected values
            expctdBoundaryOperation = 'Termination';
            expctdDirection = bdir;

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'BoundaryOperation',expctdBoundaryOperation,...
                'Direction',expctdDirection);

            % Actual values
            actualBoundaryOperation = layer.BoundaryOperation;
            actualDirection = layer.Direction;

            % Evaluation
            testCase.verifyEqual(actualBoundaryOperation,expctdBoundaryOperation);
            testCase.verifyEqual(actualDirection,expctdDirection);
        end

        function testInvalidBoundaryOperation(testCase, nchs)
            % An invalid BoundaryOperation value throws an exception

            import saivdr.dcnn.*
            testCase.verifyError(@() nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'NsoltLayer:InvalidBoundaryOperation');
        end

        function testInvalidDirection(testCase, nchs)
            % An invalid Direction value for 'Termination' throws an exception

            import saivdr.dcnn.*
            testCase.verifyError(@() nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'BoundaryOperation','Termination',...
                'Direction','InvalidDirection'),...
                'NsoltLayer:InvalidDirection');
        end

        function testPredictCircular(testCase, ...
                nchs, nrows, ncols, mus, mode, bdir, datatype)
            % 'Circular' rotates all slices by Un, regardless of Direction

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);

            % Expected values
            Un = genU.step(angles,mus);
            expctdZ = testCase.rotateDifferenceChannels_(X,nchs,Un,mode,'');

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'BoundaryOperation','Circular',...
                'Direction',bdir);

            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testPredictTermination(testCase, ...
                nchs, nrows, ncols, mus, mode, bdir, datatype)
            % Boundary slice: Za = -Ya, other slices: Za = Un*Ya
            % (Un.'*Ya in Synthesis mode) with default angles

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            pa = nchs(2);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);

            % Expected values
            Un = mus*eye(pa);
            expctdZ = testCase.rotateDifferenceChannels_(X,nchs,Un,mode,bdir);

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'Mus',mus,...
                'BoundaryOperation','Termination',...
                'Direction',bdir);

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testPredictTerminationWithRandomAngles(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % The boundary slice is rotated by -I for any Angles and Mus

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            pa = nchs(2);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            mus_ = 2*(rand(pa,1)>0.5)-1;

            % Expected values
            Un = genU.step(angles,mus_);
            expctdZ = testCase.rotateDifferenceChannels_(X,nchs,Un,mode,bdir);

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir);

            % Actual values
            layer.Mus = mus_;
            layer.Angles = angles;
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testBackwardTermination(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % dLdX: boundary slice gives -dLdZ, other slices give
            % Un.'*dLdZ (Un*dLdZ in Synthesis mode)

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            pa = nchs(2);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            mus_ = 2*(rand(pa,1)>0.5)-1;

            % Expected values
            % The adjoint of the forward rotation: Un.' in Analysis, Un in Synthesis
            Un = genU.step(angles,mus_);
            if strcmp(mode,'Analysis')
                adjointMode = 'Synthesis';
            else
                adjointMode = 'Analysis';
            end
            expctddLdX = testCase.rotateDifferenceChannels_(dLdZ,nchs,Un,adjointMode,bdir);

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir);

            % Actual values
            layer.Mus = mus_;
            layer.Angles = angles;
            [actualdLdX,~] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
        end

        function testBackwardTerminationAngleGradients(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % dLdW: the boundary slice does not contribute to the angle
            % gradients

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            ps = nchs(1);
            pa = nchs(2);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn(nAngles,1);
            mus_ = 2*(rand(pa,1)>0.5)-1;

            % Expected values
            % dLdWi = <dLdZ,(dVdWi)X> over the non-boundary slices
            mask = testCase.nonBoundaryMask_(nrows,ncols,bdir);
            expctddLdW = zeros(nAngles,1,datatype);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus_,iAngle);
                if strcmp(mode,'Synthesis')
                    dUn = dUn.';
                end
                c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols*nSamples);
                c_low = cast(dUn,datatype)*c_low;
                dVdW_X = zeros(size(X),datatype);
                dVdW_X(ps+1:ps+pa,:,:,:) = reshape(c_low,pa,nrows,ncols,nSamples);
                dVdW_X = dVdW_X.*mask;
                expctddLdW(iAngle) = sum(dLdZ.*dVdW_X,'all');
            end

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'BoundaryOperation','Termination',...
                'Direction',bdir);

            % Actual values
            layer.Mus = mus_;
            layer.Angles = angles;
            [~,actualdLdW] = layer.backward(X,[],dLdZ,[]);

            % Evaluation
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end

        function testPredictTerminationSetAfterConstruction(testCase, ...
                nchs, nrows, ncols, mode, bdir)
            % Setting BoundaryOperation and Direction after construction
            % takes effect on the next predict

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import saivdr.dictionary.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = sum(nchs);
            pa = nchs(2);
            X = randn(nChsTotal,nrows,ncols,nSamples);
            angles = randn((nChsTotal-2)*nChsTotal/8,1);
            mus_ = 2*(rand(pa,1)>0.5)-1;

            % Expected values
            Un = genU.step(angles,mus_);
            expctdZ = testCase.rotateDifferenceChannels_(X,nchs,Un,mode,bdir);

            % Instantiation of target class with 'Circular'
            % Un is generated for 'Circular' at the end of construction
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs,...
                'Mode',mode,...
                'Angles',angles,...
                'Mus',mus_);

            % Switch to 'Termination' after construction
            layer.Direction = bdir;
            layer.BoundaryOperation = 'Termination';

            % Actual values
            actualZ = layer.predict(X);

            % Evaluation
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
        end

        function testSetInvalidBoundaryOperation(testCase, nchs)
            % Setting an invalid BoundaryOperation after construction
            % throws an exception

            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs);
            testCase.verifyError(@() testCase.setProperty_(layer,...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'NsoltLayer:InvalidBoundaryOperation');
        end

        function testSetInvalidDirection(testCase, nchs)
            % Setting an invalid Direction after construction throws an
            % exception

            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs);
            testCase.verifyError(@() testCase.setProperty_(layer,...
                'Direction','InvalidDirection'),...
                'NsoltLayer:InvalidDirection');
        end

        function testPredictTerminationWithoutDirection(testCase, nchs)
            % Switching to 'Termination' without Direction after
            % construction throws an exception on predict

            % Parameters
            nrows_ = 4;
            ncols_ = 4;
            nSamples = 1;
            nChsTotal = sum(nchs);
            X = zeros(nChsTotal,nrows_,ncols_,nSamples);

            % Instantiation of target class
            import saivdr.dcnn.*
            layer = nsoltIntermediateRotation2dLayer(...
                'NumberOfChannels',nchs);
            layer.BoundaryOperation = 'Termination';

            % Evaluation
            testCase.verifyError(@() layer.predict(X),...
                'NsoltLayer:InvalidDirection');
        end

    end

    methods (Static, Access = private)

        function layer = setProperty_(layer,name,value)
            % Set a property of a layer (value object) in a function
            layer.(name) = value;
        end

        function mask = nonBoundaryMask_(nrows,ncols,bdir)
            % 1 x nRows x nCols mask which is 0 on the boundary slice
            mask = ones(1,nrows,ncols);
            if strcmp(bdir,'Horizontal')
                mask(1,:,1) = 0;   % first column
            elseif strcmp(bdir,'Vertical')
                mask(1,1,:) = 0;   % first row
            end
        end

        function Z = rotateDifferenceChannels_(X,nchs,Un,mode,bdir)
            % Rotate the difference channels by Un in Analysis mode or by
            % Un.' in Synthesis mode, and by -I on the boundary slice
            % specified by bdir ('' for no boundary slice)
            nrows_ = size(X,2);
            ncols_ = size(X,3);
            nSamples = size(X,4);
            ps = nchs(1);
            pa = nchs(2);
            if strcmp(mode,'Synthesis')
                Un = Un.';
            end
            Un = cast(Un,'like',X);
            Z = X;
            Ya = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows_*ncols_*nSamples);
            Z(ps+1:ps+pa,:,:,:) = reshape(Un*Ya,pa,nrows_,ncols_,nSamples);
            if strcmp(bdir,'Horizontal')
                Z(ps+1:ps+pa,:,1,:) = -X(ps+1:ps+pa,:,1,:);
            elseif strcmp(bdir,'Vertical')
                Z(ps+1:ps+pa,1,:,:) = -X(ps+1:ps+pa,1,:,:);
            end
        end

    end

end
