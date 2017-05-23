clear all;
close all;

%the files optimize** contain
%Grad    --- gradient, 
%Init_ts --- initial condition of forward problem
%Init_adj--- initial condition of backward problem

%additionally 
%xg - the grid
%obj- the objective function
%ic - initial condition (the one to be optimized)

figure(1);set(gca,'FontSize',18);
run('IC_OBJ.m')
plot(xg,ic,'k-','LineWidth',2);
hold on
plot(xg,obj,'r-','LineWidth',2);

run('PDEadjoint/optimize00.m')
plot(xg,Init_ts,'k*','Markersize',14);
run('PDEadjoint/optimize05.m')
plot(xg,Init_ts,'ro','Markersize',14);

xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('IC','OBJ','Iter 0','Iter 9')

figure(2);set(gca,'FontSize',18);hold on

run('PDEadjoint/optimize00.m')
plot(xg,Grad,'k*-');
run('PDEadjoint/optimize05.m')
plot(xg,Grad,'ro-');
xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('Grad at it=0','Grad at it=9')
run('fd.m')
plot(xg,gradj./Mass,'ro-');

