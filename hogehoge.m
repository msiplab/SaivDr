function value = hogehoge(img,nsolt,angs,coefs,scales)
    release(nsolt);
    set(nsolt,'Angles',angs);
    synthesizer = saivdr.dictionary.nsoltx.NsoltSynthesis2dSystem('LpPuFb2d',nsolt);
    
    diff = img - step(synthesizer,coefs,scales);
    value = sum(abs(diff(:)).^2);
end