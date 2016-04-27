function hB = fcn_build_butterfly_mtx(P,theta)
    qrtP = floor(P/4);

    hC = eye(floor(P/2));
    hS = eye(floor(P/2));
    for p = 1:qrtP
        tp = theta(p);

        hC(2*p-1:2*p, 2*p-1:2*p) = buildMtxHc_(tp);
        hS(2*p-1:2*p, 2*p-1:2*p) = buildMtxHs_(tp);
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
