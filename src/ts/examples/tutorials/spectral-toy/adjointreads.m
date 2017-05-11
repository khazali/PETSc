clear all;
close all;

run('IC_OBJ.m')
plot(xg,ic,'k-');
hold on
plot(xg,obj,'r');

run('optimize00.m')
plot(xg,Init_ts,'k*-');
run('optimize09.m')
plot(xg,Init_ts,'ro-');

xlabel('x (GLL grid)');
ylabel('f(x)- objective');

legend('IC','OBJ','Iter 1','Iter 9')