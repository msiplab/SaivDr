function value = hogehoge(img,nsolt,angs,coefs,scales)
    release(nsolt);
    set(nsolt,'Angles',angs);
    synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
    
    diff = img - step(synthesizer,coefs,scales);
    value = sum(abs(diff(:)).^2);
    
%     grad = zeros(size(angs));
%     for idx = 1:length(angs)
%         dangs = angs;
%         dangs(idx) = angs(idx)+pi/2;
%         release(nsolt);
%         set(nsolt,'Angles',dangs);
%         dsynth = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
%         
%         tmp1 = conj(diff).*step(dsynth,coefs,scales);
%         grad(idx) = -2*real(sum(tmp1(:)))/numel(tmp1);
%     end
end