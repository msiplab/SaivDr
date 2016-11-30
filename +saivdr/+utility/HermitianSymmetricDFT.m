classdef HermitianSymmetricDFT
    methods (Static = true)
        function value = hsdftmtx(nDec) %Hermitian-Symmetric DFT matrix
            value = complex(zeros(nDec));
            for u = 0:nDec-1
                for x =0:nDec-1
                    n = rem(u*(2*x+1),2*nDec);
                    value(u+1,x+1) = exp(-1i*pi*n/nDec)/sqrt(nDec);
                end
            end
        end
        
        function value = conjhsdft2(x) %conjugate-hsdft
            nDec = size(x,1);
            mtx = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(nDec);
            cmtx = conj(mtx);
            value = (cmtx*(cmtx*x).').';
        end
        
        function value = conjihsdft2(x) %conjugate-inverse hsdft
            nDec = size(x,1);
            mtx = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(nDec);
            value = (mtx.'*(mtx.'*x).').';
        end
    end
end