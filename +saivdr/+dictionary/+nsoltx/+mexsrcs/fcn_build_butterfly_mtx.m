function hB = fcn_build_butterfly_mtx(P,theta)
    qtrP = floor(P/4);
	
    hC = [];
    hS = [];
    for p = 1:qrtP
        tp = theta(p);
        
        hC = blkdiag(hC, _buildMtxHc(tp));
        hS = blkdiag(hS, _buildMtxHs(tp));
    end
	
    if odd(qtrP)
        hC = blkdiag(hC, 1);
        hS = blkdiag(hS, 1);
    end
    
    hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
end

function mtxHc = _buildMtxHc(t)
    mtxHc =     [-1i*cos(t), -1i*sin(t);
                cos(t) , -sin(t)];
end

function mtxHs = _buildMtxHs(t)
    mtxHs =     [ -1i*sin(t), -1i*cos(t);
                sin(t) , -cos(t)];
end
