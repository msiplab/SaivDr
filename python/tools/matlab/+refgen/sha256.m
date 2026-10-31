function h = sha256(file)
%SHA256 SHA-256 digest of a file as a lowercase hexadecimal string
fid = fopen(file,'r');
bytes = fread(fid,inf,'*uint8');
fclose(fid);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
