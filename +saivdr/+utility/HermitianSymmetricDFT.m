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
        
        function value = hsdft(v)
            mtx = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(size(v,1));
            value = mtx*v;
        end
        
        function value = ihsdft(v)
            mtx = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(size(v,1));
            value = mtx'*v;
        end
        
        function value = hsdft2(v) %hsdft
            [decY,decX] = size(v);
            mtxY = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decY);
            mtxX = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decX);
            value = (mtxX*(mtxY*v).').';
        end
        
        function value = ihsdft2(v) %inverse hsdft
            [decY,decX] = size(v);
            mtxY = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decY);
            mtxX = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decX);
            value = (mtxX'*(mtxY'*v).').';
        end
        
%         function value = conjhsdft2(v) %conjugate-hsdft
%             [decY,decX] = size(v);
%             mtxY = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decY);
%             mtxX = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decX);
%             value = (conj(mtxX)*(conj(mtxY)*v).').';
%         end
%         
%         function value = conjihsdft2(v) %conjugate-inverse hsdft
%             [decY,decX] = size(v);
%             mtxY = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decY);
%             mtxX = saivdr.utility.HermitianSymmetricDFT.hsdftmtx(decX);
%             value = (mtxX.'*(mtxY.'*v).').';
%         end
    end
end