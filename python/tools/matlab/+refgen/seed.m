function s = seed(name)
%SEED Deterministic random seed derived from a case name
v = double(char(name));
s = mod(sum(v.*(1:numel(v))) + 7919*numel(v),2^31-1);
end
