classdef nsoltIntermediateRotation2dLayerBoundaryOperationTestCase < matlab.unittest.TestCase
    %NSOLTINTERMEDIATEROTATION2DLAYERBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of nsoltIntermediateRotation2dLayer.
    %
    %   Termination replaces the rotation of the boundary slice
    %   (first column for horizontal, first row for vertical) with -I and
    %   excludes it from the angle gradients, as in
    %   saivdr.dictionary.nsoltx.NsoltAtomExtender2d.
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
        bdir = { 'Horizontal', 'Vertical' }; % TODO: property name to be decided
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            % TODO: checkLayer for a layer with 'BoundaryOperation','Termination'
        end
    end

    methods (Test)

        function testConstructorDefaultBoundaryOperation(testCase, nchs)
            % Default BoundaryOperation is 'Circular' (backward compatibility)
            testCase.verifyFail('Not implemented yet.');
        end

        function testConstructorWithBoundaryOperation(testCase, nchs, bdir)
            % BoundaryOperation and boundary direction are set by the constructor
            testCase.verifyFail('Not implemented yet.');
        end

        function testInvalidBoundaryOperation(testCase, nchs)
            % An invalid BoundaryOperation value throws an exception
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictCircular(testCase, ...
                nchs, nrows, ncols, mus, mode, datatype)
            % 'Circular' gives the same output as the current layer
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictTermination(testCase, ...
                nchs, nrows, ncols, mus, mode, bdir, datatype)
            % Boundary slice: Za = -Ya, other slices: Za = Un*Ya (Un.'*Ya in Synthesis)
            testCase.verifyFail('Not implemented yet.');
        end

        function testPredictTerminationWithRandomAngles(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % The boundary slice is rotated by -I for any Angles and Mus
            testCase.verifyFail('Not implemented yet.');
        end

        function testBackwardTermination(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % dLdX: boundary slice gives -dLdZ, other slices give Un.'*dLdZ
            % (Un*dLdZ in Synthesis mode)
            testCase.verifyFail('Not implemented yet.');
        end

        function testBackwardTerminationAngleGradients(testCase, ...
                nchs, nrows, ncols, mode, bdir, datatype)
            % dLdW: the boundary slice does not contribute to the angle gradients
            testCase.verifyFail('Not implemented yet.');
        end

    end

end
