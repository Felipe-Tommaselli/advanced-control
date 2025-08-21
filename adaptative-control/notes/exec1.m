close all
clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dados de simulacao
% \dot{x}(t) = -2x(t)+bu(t)
% x = (b/(s+2))u

%%%%%%%%%%%%%%%%%%%%%%%%%%%

b     = 3;
num   = b;
den   = [1 2];
ftx   = tf(num,den);
N     = 3;
dt    = 0.01;
T     = 0:dt:N;
u     = ones(1,N/dt+1); 
[Y,T] = lsim(ftx,u,T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% algoritmo de identificacao de parametros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

teta(1)     = 0;
erro(1)     = 0;
ftx1        = tf(1,den);
[phi,T]     = lsim(ftx1,u,T);
gama        = 1000;
ms          = 10;

for k = 1:N/dt
    
    teta(k+1) = teta(k) + gama*dt*((Y(k) - teta(k)*phi(k))/ms)*phi(k);
    erro(k+1)   = Y(k) - teta(k)*phi(k);
end

figure(1)
plot(T,teta,'LineWidth',2)
hold on
plot(T,erro,'LineWidth',2)
legend('Parametro b','Erro')
xlabel('amostras')
ylabel('Parametro Identificado')
ax =gca;
ax.FontSize = 14;
hold off
saveas(1,'ex1_agosto_2021_a')

% Gráfico do erro
figure(2)
plot(T,erro,'linewidth',2)
xlabel('samples')
ylabel('erro')
ax

