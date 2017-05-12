clear all;
%close all;


figure;set(gca,'FontSize',18);
run('IC_OBJ.m')
plot(xg,ic,'k-','LineWidth',2);
hold on
plot(xg,obj,'r-','LineWidth',2);

run('optimize00.m')
plot(xg,Init_ts,'k*','Markersize',14);
run('optimize06.m')
plot(xg,Init_ts,'ro','Markersize',14);

xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('IC','OBJ','Iter 0','Iter 9')


figure;set(gca,'FontSize',18);hold on

run('optimize00.m')
plot(xg,Grad,'k*-');
run('optimize06.m')
plot(xg,Grad,'ro-');

