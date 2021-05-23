data = dlmread('swissroll.dat');
chunksize = 1/5;
k = 5;

% Divide samples into cubes
limits = 0:0.2:1;
cubes = cell(5,5,5);
for i=1:1000
  s = data(i,:);
  pos = limits-s(1);
  x = find((limits-s(1))>0, 1, 'first') - 1;
  y = find((limits-s(2))>0, 1, 'first') - 1;
  z = find((limits-s(3))>0, 1, 'first') - 1;
  
  cubes(x,y,z) = cat(1,cubes(x,y,z){1},s);
  
endfor

% Check if ok
tot = 0
ii=0
for i=1:5*5*5
  s = size(cubes(i){1})(1);
  tot += s;
  fprintf('%i , %i\n', i, s)
endfor

% Perform PCA for each cube
povs_c = cell(5,5,5);
means = cell(5,5,5);
evalues_c = cell(5,5,5);
for i=1:5*5*5
  if size(cubes(i){1})(1) < 5
    continue
  endif
  x = cubes(i){1};
  y = x - mean(x);
  C = cov(y);
  evalues = eig(C);
  fprintf('%d, eval sum=%d\n',i, sum(evalues))
  evalues_c(i) = cat(1, povs_c(i){1}, evalues);
  povs_c(i) = cat(1, povs_c(i){1}, cumsum(evalues)/sum(evalues));
  means(i) = mean(x);
endfor

% Report results
for x=1:5
  for y=1:5
    for z=1:5
      %fprintf('%d, %d, %d\n', x,y,z)
      if size(povs_c(x,y,z){1})(1) == 0
        continue
      endif
      povs = povs_c(x,y,z){1};
      ev_needed = find(povs>=0.9, 1, 'first')
      fprintf('Cube location:\n')
      fprintf(' X: %d - %d\n',limits(x),limits(x+1))
      fprintf(' Y: %d - %d\n',limits(y),limits(y+1))
      fprintf(' Z: %d - %d\n',limits(z),limits(z+1))
      fprintf('Eigenvalues needed: %i\n', ev_needed)
     printf('povs: %d, %d, %d\n\n', povs)
      #fprintf('Eigenvalues needed: %i\n', ev_needed)
      
    endfor
  endfor
endfor
