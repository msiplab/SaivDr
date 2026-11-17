function s = gendegradation(c)
%GENDEGRADATION Forward and adjoint processes and the maximum eigenvalue
args = namedargs2cell(c.params);
linproc = feval(['saivdr.degradation.linearprocess.' c.class],args{:},'DataType','Image');
x = rand(c.size);
z = step(linproc,x);
adjproc = clone(linproc);
set(adjproc,'ProcessingMode','Adjoint');
y = randn(size(z));
s.meta_class = c.class;
s.meta_params = jsonencode(c.params);
s.x = x;
s.x_size = refgen.padsize(x,4);
s.z = z;
s.z_size = refgen.padsize(z,4);
s.y = y;
s.y_size = refgen.padsize(y,4);
s.adjoint = step(adjproc,y);
s.adjoint_size = refgen.padsize(s.adjoint,4);
s.lambda_max = get(linproc,'LambdaMax');
if strcmp(c.class,'PixelLossSystem')
    s.mask = double(step(clone(linproc),ones(c.size(1:2))) ~= 0);
end
end
