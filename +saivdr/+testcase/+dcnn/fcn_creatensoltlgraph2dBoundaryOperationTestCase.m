classdef fcn_creatensoltlgraph2dBoundaryOperationTestCase < matlab.unittest.TestCase
    %FCN_CREATENSOLTLGRAPH2DBOUNDARYOPERATIONTESTCASE
    %
    %   Test cases for the BoundaryOperation option ('Circular' | 'Termination')
    %   of saivdr.dcnn.fcn_creatensoltlgraph2d, and for copying it by
    %   saivdr.dcnn.fcn_cpparamssyn2ana.
    %
    %   'Termination' is given to the odd-order intermediate rotation
    %   layers, i.e., those right after the shift of the difference
    %   channels, with Direction 'Horizontal' (Vh) or 'Vertical' (Vv) in
    %   both the analyzer and the synthesizer. The even-order layers stay
    %   'Circular'.
    %
    %   As a result, the coefficients never wrap around to the opposite
    %   boundary, while the perfect reconstruction is kept.
    %
    %   The networks are evaluated in double precision. Since dlnetwork
    %   casts the learnable parameters to single, the random angles are
    %   given in single, copied to the analyzer, and then both networks
    %   are cast to double so that they are exactly adjoint.
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
    % Requirements: MATLAB R2021a
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

    properties (Constant)
        NBLOCKS = [8 8]      % Number of blocks [vertical horizontal]
        DECFACTOR = [2 2]    % Decimation factor [vertical horizontal]
        STDANG = pi/6        % Standard deviation of random angles
        TOL = 1e-10          % Absolute tolerance in double precision
    end

    properties (TestParameter)
        nchs = { [2 2], [3 3] };
        ppOrder = { [2 2], [4 4], [0 2], [2 0] };
        boundaryOperation = { 'Circular', 'Termination' };
    end

    methods (Test)

        function testDefaultBoundaryOperation(testCase, nchs, ppOrder)
            % All intermediate rotation layers are 'Circular' by default

            % Instantiation of target
            lgraph = testCase.createLgraph_(nchs,ppOrder,'Whole');

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            for iLayer = 1:numel(layers)
                testCase.verifyEqual(layers(iLayer).BoundaryOperation,'Circular',...
                    layers(iLayer).Name);
            end
        end

        function testLayerSettings(testCase, nchs, ppOrder, boundaryOperation)
            % Odd-order Vh/Vv layers get the given BoundaryOperation with
            % Direction 'Horizontal'/'Vertical'. Even-order ones stay
            % 'Circular'

            % Instantiation of target
            lgraph = testCase.createLgraph_(nchs,ppOrder,'Whole',...
                'BoundaryOperation',boundaryOperation);

            % Evaluation
            layers = testCase.intermediateRotationLayers_(lgraph);
            nExpctdLayers = 2*(ppOrder(1)+ppOrder(2));
            testCase.verifyNumElements(layers,nExpctdLayers);
            for iLayer = 1:numel(layers)
                layer = layers(iLayer);
                tokens = regexp(layer.Name,'_V([hv])(\d+)~?$','tokens','once');
                order = str2double(tokens{2});
                if mod(order,2) == 1
                    expctdBoundaryOperation = boundaryOperation;
                    if strcmp(tokens{1},'h')
                        expctdDirection = 'Horizontal';
                    else
                        expctdDirection = 'Vertical';
                    end
                else
                    expctdBoundaryOperation = 'Circular';
                    expctdDirection = '';
                end
                testCase.verifyEqual(layer.BoundaryOperation,...
                    expctdBoundaryOperation,layer.Name);
                if strcmp(expctdBoundaryOperation,'Termination')
                    testCase.verifyEqual(layer.Direction,...
                        expctdDirection,layer.Name);
                end
            end
        end

        function testInvalidBoundaryOperation(testCase)
            % An invalid BoundaryOperation throws an exception
            testCase.verifyError(@() testCase.createLgraph_([2 2],[2 2],'Whole',...
                'BoundaryOperation','InvalidBoundaryOperation'),...
                'NsoltLayer:InvalidBoundaryOperation');
        end

        function testPerfectReconstruction(testCase, ...
                nchs, ppOrder, boundaryOperation)
            % The synthesizer reconstructs the input of the analyzer

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                nchs,ppOrder,boundaryOperation,boundaryOperation);

            % Evaluation
            [actualY,X] = testCase.reconstruct_(analysisnet,synthesisnet,...
                testCase.NBLOCKS.*testCase.DECFACTOR);
            testCase.verifyClass(actualY,'double');
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

        function testNoWrapAroundHorizontalTermination(testCase, ...
                nchs, ppOrder)
            % With 'Termination', an input on the left (right) edge blocks
            % never reaches the coefficients of the right (left) edge blocks

            testCase.assumeGreaterThan(ppOrder(2),0,...
                'No horizontal overlap.')
            [leftToRight,rightToLeft] = testCase.edgeResponses_(...
                nchs,ppOrder,'Termination',2);

            % Evaluation
            testCase.verifyEqual(leftToRight,0,'AbsTol',testCase.TOL,...
                'Left edge input wraps around to the right edge.');
            testCase.verifyEqual(rightToLeft,0,'AbsTol',testCase.TOL,...
                'Right edge input wraps around to the left edge.');
        end

        function testNoWrapAroundVerticalTermination(testCase, ...
                nchs, ppOrder)
            % With 'Termination', an input on the top (bottom) edge blocks
            % never reaches the coefficients of the bottom (top) edge blocks

            testCase.assumeGreaterThan(ppOrder(1),0,...
                'No vertical overlap.')
            [topToBottom,bottomToTop] = testCase.edgeResponses_(...
                nchs,ppOrder,'Termination',1);

            % Evaluation
            testCase.verifyEqual(topToBottom,0,'AbsTol',testCase.TOL,...
                'Top edge input wraps around to the bottom edge.');
            testCase.verifyEqual(bottomToTop,0,'AbsTol',testCase.TOL,...
                'Bottom edge input wraps around to the top edge.');
        end

        function testWrapAroundCircular(testCase, nchs, ppOrder)
            % With 'Circular', an input on an edge reaches the opposite
            % edge in the overlapping direction (sensitivity check of the
            % no-wrap-around tests)

            for dim = find(ppOrder > 0)
                [fwd,bwd] = testCase.edgeResponses_(...
                    nchs,ppOrder,'Circular',dim);
                testCase.verifyGreaterThan(fwd,1e-4,...
                    sprintf('No wrap-around in dimension %d.',dim));
                testCase.verifyGreaterThan(bwd,1e-4,...
                    sprintf('No wrap-around in dimension %d.',dim));
            end
        end

        function testCopyParamsSyn2AnaBoundaryOperation(testCase, ...
                nchs, ppOrder, boundaryOperation)
            % fcn_cpparamssyn2ana copies BoundaryOperation and Direction
            % so that the analyzer created with the other mode becomes
            % the adjoint of the synthesizer

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(testCase.TOL);
            otherOperation = testCase.otherOperation_(boundaryOperation);

            % Instantiation of target
            [analysisnet,synthesisnet] = testCase.createNetworks_(...
                nchs,ppOrder,otherOperation,boundaryOperation);

            % Evaluation of the layer settings
            testCase.verifyCopiedSettings_(...
                layerGraph(synthesisnet),layerGraph(analysisnet));

            % Evaluation of the perfect reconstruction
            [actualY,X] = testCase.reconstruct_(analysisnet,synthesisnet,...
                testCase.NBLOCKS.*testCase.DECFACTOR);
            testCase.verifyThat(actualY,IsEqualTo(X,'Within',tolObj));
        end

    end

    methods (Access = private)

        function lgraph = createLgraph_(testCase,nchs,ppOrder,mode,varargin)
            import saivdr.dcnn.*
            lgraph = fcn_creatensoltlgraph2d([],...
                'InputSize',testCase.NBLOCKS.*testCase.DECFACTOR,...
                'NumberOfChannels',nchs,...
                'DecimationFactor',testCase.DECFACTOR,...
                'PolyPhaseOrder',ppOrder,...
                'NumberOfLevels',1,...
                'Mode',mode,...
                varargin{:});
        end

        function [analysisnet,synthesisnet] = createNetworks_(testCase,...
                nchs,ppOrder,anaOperation,synOperation)
            % Synthesizer with random angles and the analyzer with the
            % angles (and the boundary operation) copied from the
            % synthesizer
            import saivdr.dcnn.*
            analysislgraph = testCase.createLgraph_(nchs,ppOrder,...
                'Analyzer','BoundaryOperation',anaOperation); %#ok<NASGU> used in evalc
            synthesislgraph = testCase.createLgraph_(nchs,ppOrder,...
                'Synthesizer','BoundaryOperation',synOperation);
            % Random angles in single, since dlnetwork casts learnables to single
            synthesisnet = dlnetwork(synthesislgraph);
            isAngles = synthesisnet.Learnables.Parameter == "Angles";
            synthesisnet.Learnables.Value(isAngles) = ...
                cellfun(@(x) x+testCase.STDANG*randn(size(x),'like',x), ...
                synthesisnet.Learnables.Value(isAngles),'UniformOutput',false);
            % evalc suppresses the messages of fcn_cpparamssyn2ana
            [~,analysislgraph] = evalc(...
                'fcn_cpparamssyn2ana(analysislgraph,layerGraph(synthesisnet))');
            analysisnet = dlnetwork(analysislgraph);
            % Cast both networks to double after copying the angles
            analysisnet = dlupdate(@double,analysisnet);
            synthesisnet = dlupdate(@double,synthesisnet);
        end

        function [fwd,bwd] = edgeResponses_(testCase,nchs,ppOrder,...
                boundaryOperation,dim)
            % fwd: max |coefs| on the last edge blocks for an input on the
            %      first edge blocks along dimension dim
            % bwd: max |coefs| on the first edge blocks for an input on the
            %      last edge blocks along dimension dim
            szInput = testCase.NBLOCKS.*testCase.DECFACTOR;
            analysisnet = testCase.createNetworks_(nchs,ppOrder,...
                boundaryOperation,boundaryOperation);
            edgeWidth = testCase.DECFACTOR(dim);
            %
            X = zeros(szInput);
            idx = {':',':'};
            idx{dim} = 1:edgeWidth;
            X(idx{:}) = randn(size(X(idx{:})));
            fwd = testCase.edgeCoefMax_(analysisnet,X,dim,'last');
            %
            X = zeros(szInput);
            idx = {':',':'};
            idx{dim} = szInput(dim)-edgeWidth+1:szInput(dim);
            X(idx{:}) = randn(size(X(idx{:})));
            bwd = testCase.edgeCoefMax_(analysisnet,X,dim,'first');
        end

        function verifyCopiedSettings_(testCase,synthesislgraph,analysislgraph)
            % The intermediate rotation layers of the analyzer have the
            % same BoundaryOperation and Direction as those of the
            % synthesizer
            synLayers = testCase.intermediateRotationLayers_(synthesislgraph);
            anaNames = string({analysislgraph.Layers.Name});
            for iLayer = 1:numel(synLayers)
                synLayer = synLayers(iLayer);
                anaName = erase(string(synLayer.Name),"~");
                anaLayer = analysislgraph.Layers(anaNames == anaName);
                testCase.verifyEqual(anaLayer.BoundaryOperation,...
                    synLayer.BoundaryOperation,anaName);
                testCase.verifyEqual(anaLayer.Direction,...
                    synLayer.Direction,anaName);
            end
        end

    end

    methods (Static, Access = private)

        function [Y,X] = reconstruct_(analysisnet,synthesisnet,szInput)
            % Analysis followed by synthesis of a random input in double
            X = randn(szInput);
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
            Y = extractdata(synthesisnet.predict(s{:}));
        end

        function value = edgeCoefMax_(analysisnet,X,dim,edge)
            % Maximum absolute value of the AC and DC coefficients on the
            % first or last blocks along dimension dim
            s = cell(1,2);
            [s{1:2}] = analysisnet.predict(dlarray(X,'SSCB'));
            value = 0;
            for iCoef = 1:2
                coefs = extractdata(s{iCoef});
                idx = {':',':',':',':'};
                if strcmp(edge,'first')
                    idx{dim} = 1;
                else
                    idx{dim} = size(coefs,dim);
                end
                value = max(value,max(abs(coefs(idx{:})),[],'all'));
            end
        end

        function layers = intermediateRotationLayers_(lgraph)
            isTarget = arrayfun(@(x) ...
                isa(x,'saivdr.dcnn.nsoltIntermediateRotation2dLayer'),...
                lgraph.Layers);
            layers = lgraph.Layers(isTarget);
        end

        function value = otherOperation_(boundaryOperation)
            if strcmp(boundaryOperation,'Circular')
                value = 'Termination';
            else
                value = 'Circular';
            end
        end

    end

end
