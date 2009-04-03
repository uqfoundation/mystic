%
% see test_circle.py
% solves the dual problem,  matlab implementation
% (a python version would be nice, but needs a quadprog solver)
%
% Patrick Hung

randseed(123);

x0 = 10;
y0 = 20;
R0 = 3;

npt = 30;

xy = [];
% generate the points
while 1
   pt = rand(1,2) *2-1;
   if norm(pt) <= 1 
      xy = [xy;pt(1)*R0+x0 pt(2)*R0+y0];
   end
   if size(xy,1) == npt 
      break
   end
end

theta = [0:0.01:2*pi];

Q = zeros(npt,npt);
for i = 1:npt
 for j = 1:npt
   % super dumb. don't do this at home
   Q(i,j) = dot(xy(i,:),xy(j,:));
 end
end
p = -diag(Q);

% sets up quadratic program parameters
H = 2*Q;
f = p;
A = eye(npt);
b = ones(npt,1);

al = quadprog(H,f,[],[], ones(1,npt),1, zeros(npt,1),ones(npt,1));

center = al'*xy;
sv = find(al>0.001);
R = norm(xy(sv(1),:)-center);

clf
plot(xy(:,1),xy(:,2),'k.')
hold on
plot(xy(sv,1),xy(sv,2),'ro')
axis equal
plot(R0 *cos(theta)+x0, R0*sin(theta)+y0,'r-')
plot(R *cos(theta)+center(1), R*sin(theta)+center(2),'g-')
plot(center(1),center(2),'k+')
plot([center(1) xy(sv(1),1)],[center(2) xy(sv(1),2)],'k--')

% end of file
