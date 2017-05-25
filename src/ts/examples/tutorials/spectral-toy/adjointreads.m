clear all;
close all;

fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';


%the files optimize** contain
%Grad    --- gradient, 
%Init_ts --- initial condition of forward problem
%Init_adj--- initial condition of backward problem

%additionally 
%xg - the grid
%obj- the objective function
%ic - initial condition (the one to be optimized)

figure(1);
run('IC_OBJ.m')
plot(xg,ic,'k-','LineWidth',2,'Markersize',12);
hold on
plot(xg,obj,'r-','LineWidth',2,'Markersize',12);
xlabel('x (GLL grid)');
ylabel('Objective/Initial condition');
legend('IC','OBJ')
axis tight; axis square
set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)

figure(3)
run('PDEadjoint/optimize00.m')
plot(xg,Init_ts,'k*','Markersize',10); drawnow
hold on
for ii=1:30
file=sprintf('PDEadjoint/optimize%02d.m',ii);  
run(file)
plot(xg,Init_ts,'go','Markersize',10); drawnow;
%keyboard
end
xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('IC','OBJ','Iter 0','Iter 9')

figure(2);set(gca,'FontSize',18);hold on

run('PDEadjoint/optimize00.m')
plot(xg,Grad,'k*-');
run('PDEadjoint/optimize10.m')
plot(xg,Grad,'ro-');

set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)

xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('Grad at it=0','Grad at it=9')

figure(10)
run('fd.m')
%plot(gradj)
plot(xg,gradj./Mass,'ro-','LineWidth',2,'Markersize',12);
hold on
run('PDEadjoint/optimize00.m')
plot(xg,Grad,'k*-','LineWidth',2,'Markersize',10);

set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)

legend('Gradient FD','Gradient Adjoint')
xlabel('x (GLL grid)');
ylabel('Gradient');
axis tight; axis square

errgrad=max(abs(gradj./Mass-Grad))
break

figure(21)
semilogy(1:31,TAO,'b','LineWidth',2)
hold on
semilogy(1:31,L2,'r','LineWidth',2)
grid on
xlabel('No iterations');
ylabel('Cost function');

legend('TAO','User')
