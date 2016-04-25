function hB = fcn_build_butterfly_mtx(P,theta)
    qrtP = floor(P/4);
	
    hC = [];
    hS = [];
    for p = 1:qrtP
        tp = theta(p);
        
        hC = blkdiag(hC, buildMtxHc_(tp));
        hS = blkdiag(hS, buildMtxHs_(tp));
    end
	
    if mod(qrtP,2) == 1
        hC = blkdiag(hC, 1);
        hS = blkdiag(hS, 1);
    end
    
    hB = [hC, conj(hC); 1i*hS, -1i*conj(hS)]/sqrt(2);
end

function mtxHc = buildMtxHc_(t)
    mtxHc =     [-1i*cos(t), -1i*sin(t);
                cos(t) , -sin(t)];
end

function mtxHs = buildMtxHs_(t)
    mtxHs =     [ -1i*sin(t), -1i*cos(t);
                sin(t) , -cos(t)];
end
