clc, clear, close all;

% code switches, (1 == on, 0 == off)
delay_com = 1; % communication delay
delay_up = 1; % update delay
delay_meas = 1; % measurement delay

% input parameters
N = 10; % number of agents
part_max = 2; % max dimension of an agent's input partition
outSize = 1; % agents' output size
numFun = 10; % number of switching functions
uMin = -10; % input lower bound
uMax = 10; % input upper bound
iter = 1000; % number of iterations allowed per switching function
gamma = 1e-3; % step size

% assigning size of each agent's partition
for i = 1:N
   n(i) = randi(part_max);
   n(i) = part_max;
end
sum(n)

nn = sum(n);
mm = outSize*N;
C = rand(mm,sum(n));

%% Delay Probabilities
prob = 0.01;

% communication delay
if delay_com == 1
    com_prob = ones(N,1)*prob; % generates probability of communication each iteration
else
    com_prob = ones(N,1);
end

% computation delay
if delay_up == 1
    up_prob = ones(N,1)*prob;
else
    up_prob = ones(N,1);
end

% probability of measurement
if delay_meas == 1
    measProb = ones(N,1)*prob;
else
    measProb = ones(N,1);
end
%% Building Objective Functions

% constructing PD matricies for QP switching
Q = zeros(nn,nn,numFun);
P = zeros(mm,mm,numFun);

for i = 1:numFun
    Q(:,:,i) = rand(nn,nn);
    Q(:,:,i) = Q(:,:,i)*Q(:,:,i)' / 5;
    Q(:,:,i) = Q(:,:,i) + diag(20*rand(nn,1));
    
    P(:,:,i) = rand(mm,mm);
    P(:,:,i) = P(:,:,i)*P(:,:,i)' / 5;
    P(:,:,i) = P(:,:,i) + diag(20*rand(mm,1));
    
    R(:,:,i) = Q(:,:,i) + C'*P(:,:,i)*C;
    
    
end

% Linear terms for objective function
q = -20*rand(nn,numFun);
p = -20*rand(mm,numFun);
r = q + C'*p; % linear term of transformed objective

%Quadprog Parameters
options = optimset('Display', 'off');
b = zeros(nn,1); % Constraint vector
A = zeros(nn); % Constraint matrix
ub = uMax*ones(nn,1); % upper bound of constraint set size
lb = -ub; % lower bound
u0 = zeros(nn,1); % initial guess for fmincon


% Optimization with switching functions
avgCost = zeros(numFun*iter,1);
agent_cost = zeros(numFun*iter,N);
agent_error = zeros(numFun*iter,N);
error = zeros(numFun*iter,1);

BB = [5;25;50];
for ii = 1:length(BB) 
    %% Simulation
    counter = 1;
    yi = rand(mm,N); % stores agents' local output vector
    ui = rand(sum(n),N); % stores agents' local input vector
    ss = zeros(nn,iter*numFun); % store s^{t}(k)
    qq = zeros(mm,iter*numFun); % store q(k)

    lastUp = zeros(N,1); % keeps track most recent update time
    lastCom = zeros(N,1); % keeps track most recent communication time
    lastMeas = zeros(N,1); % keeps track most recent measurement time

    % Define true state of network
    for k = 1:N
        ind1 = sum(n(1:k - 1)) + 1;
        ind2 = ind1 + n(k) - 1;
        uTrue(ind1:ind2,1) = ui(ind1:ind2,k)';
    end
    yTrue = C*uTrue; % true state of network outputs
    JstarVec = [];

    
    for i = 1:numFun % Function switching loop
    
        uStar(:,i) = quadprog(R(:,:,i),r(:,i),A,b,A,b,lb,ub,u0,options);
        yStar(:,i) = C*uStar(:,i);
        Jstar(i) = (1/2)*uStar(:,i)'*R(:,:,i)*uStar(:,i) + r(:,i)'*uStar(:,i);

        % Asynchronous agent update loop
        for j = 1:iter


            % UPDATE
            % dfdu = R(:,:,i)*ui + r(:,i);
            dfdu = Q(:,:,i)*ui + q(:,i) + C'*(P(:,:,i)*yi + p(:,i));

            uNew = ui - gamma * dfdu;
            uNew(find(uNew<uMin)) = uMin; % projection onto constraint set
            uNew(find(uNew>uMax)) = uMax; % projection onto constraint set

            % updating local decision vectors
            up_reality = rand(N,1);
            update = find(up_reality < up_prob);

            if ~isempty(update)
                for k = 1:size(update,1)
                    agentUp = update(k);
                    ind1 = sum(n(1:agentUp - 1)) + 1;
                    ind2 = ind1 + n(agentUp) - 1;
                    ss(ind1:ind2,counter) = uNew(ind1:ind2,agentUp) - ui(ind1:ind2,agentUp);
                    ui(ind1:ind2,agentUp) = uNew(ind1:ind2,agentUp);
                    lastUp(agentUp) = counter;
                end
            end

            % update agents if the max bounded delay is met
            update = find(lastUp < counter-BB(ii)+1);
            if ~isempty(update)
                for k = 1:size(update,1)
                    agentUp = update(k);
                    ind1 = sum(n(1:agentUp - 1)) + 1;
                    ind2 = ind1 + n(agentUp) - 1;
                    ss(ind1:ind2,counter) = uNew(ind1:ind2,agentUp) - ui(ind1:ind2,agentUp);
                    ui(ind1:ind2,agentUp) = uNew(ind1:ind2,agentUp);
                    lastUp(agentUp) = counter;
                end
            end

            % MEASURE OUTPUTS
            measReality = rand(N,1);
            measure = find(measReality < measProb);
            if ~isempty(measure)
                for k = 1:size(measure,1)
                    agentMeas = measure(k);
                    ind3 = agentMeas*outSize - (outSize-1);
                    ind4 = agentMeas*outSize;
                    qq(ind3:ind4,counter) = yTrue(ind3:ind4) - yi(ind3:ind4,agentMeas);
                    yi(ind3:ind4,agentMeas) = yTrue(ind3:ind4);
                    lastMeas(agentMeas) = counter;
                end 
            end

            % measure if max measurement delay bound is met
            measure = find(lastMeas < counter-BB(ii)+1);
            if ~isempty(measure)
                for k = 1:size(measure,1)
                    agentMeas = measure(k);
                    ind3 = agentMeas*outSize - (outSize-1);
                    ind4 = agentMeas*outSize;
                    qq(ind3:ind4,counter) = yTrue(ind3:ind4) - yi(ind3:ind4,agentMeas);
                    yi(ind3:ind4,agentMeas) = yTrue(ind3:ind4);
                    lastMeas(agentMeas) = counter;
                end 
            end

            % UPDATE TRUE STATE
            for k = 1:N
                ind1 = sum(n(1:k - 1)) + 1;
                ind2 = ind1 + n(k) - 1;
                uTrue(ind1:ind2,1) = ui(ind1:ind2,k)';
            end
            yTrue = C*uTrue; % true state of network outputs


            % COMMUNICATE
            com_reality = rand(N,1);
            communicate = find(com_reality < com_prob);

            % communicating if communicate is not empty
            if ~isempty(communicate)
                for k = 1:size(communicate,1)
                    agentCom = communicate(k);
                    ind1 = sum(n(1:agentCom - 1)) + 1;
                    ind2 = ind1 + n(agentCom) - 1;
                    ind3 = agentCom*outSize - (outSize-1);
                    ind4 = agentCom*outSize;
                    ui(ind1:ind2,:) = repmat(ui(ind1:ind2,agentCom),1,N);
                    yi(ind3:ind4,:) = repmat(yi(ind3:ind4,agentCom),1,N);
                    lastCom(agentCom) = counter;
                end
            end

            % communicate if max communication delay is met
            communicate = find(lastCom < counter-BB(ii)+1);
            if ~isempty(communicate)
                for k = 1:size(communicate,1)
                    agentCom = communicate(k);
                    ind1 = sum(n(1:agentCom - 1)) + 1;
                    ind2 = ind1 + n(agentCom) - 1;
                    ind3 = agentCom*outSize - (outSize-1);
                    ind4 = agentCom*outSize;
                    ui(ind1:ind2,:) = repmat(ui(ind1:ind2,agentCom),1,N);
                    yi(ind3:ind4,:) = repmat(yi(ind3:ind4,agentCom),1,N);
                    lastCom(agentCom) = counter;
                end
            end


            % COST & ERROR CALCULATION

            % agent cost
            for k = 1:N
                agent_cost(counter,k) = (1/2) * ui(:,k)' * R(:,:,i) * ui(:,k) + r(:,i)' * ui(:,k);
            end
            avgCost(counter) = sum(agent_cost(counter,:))/N; % average cost across agents

            % agent error
            for k = 1:N
                agent_error(counter,k) = norm([ui(:,k); yi(:,k)] - [uStar(:,i); yStar(:,i)]);
                u_error(counter,k) = sum(norm(ui(:,k) - uStar(:,i)));
                y_error(counter,k) = sum(norm(yi(:,k) - yStar(:,i)));
            end
            error(counter) = sum(agent_error(counter,:)); % total error

            % true state cost
            trueCost(counter,ii) = (1/2) * uTrue'*Q(:,:,i)*uTrue + q(:,i)'*uTrue + (1/2) *yTrue'*P(:,:,i)*yTrue + p(:,i)'*yTrue;

            counter = counter + 1;
        end

        JstarVec = [JstarVec; ones(iter,1)*Jstar(i)];

    end
    BB_error(ii,:) = error;
    alpha(:,ii) = trueCost(:,ii) - JstarVec;
    
    for k = BB(ii)+1:counter-1
        sum1 = 0;
        sum2 = 0;
        for tau = k-BB(ii):k-1
            sum1 = sum1 + norm(ss(:,tau))^2;
            sum2 = sum2 + norm(qq(:,tau))^2;
        end
        beta(k,ii) = sum1;
        delta(k,ii) = sum2;
    end
end
%%
% Calculating alpha, beta, delta

% for k = BB(ii)+1:counter-1
%     sum = 0;
%     sum2 = 0;
%     for tau = k-BB(ii):k-1
%         sum = sum + norm(ss(:,tau))^2;
%         sum2 = sum2 + norm(qq(:,tau))^2;
%     end
%     beta(k) = sum;
%     delta(k) = sum2;
%     
% end
% deltaUB = BB(ii)*mm*norm(C)^2*beta;


t = linspace(1,numFun*iter, numFun*iter);

figure(1)
plot(t,BB_error(2,:));
xlabel('Iterations (k)','interpreter','latex', 'FontSize', 16);
ylabel('$\sum^{N}_{i=1} \| (x^{i}(k), y^{i}(k))-(x^*(t_\ell), y^*(t_\ell))\|_{2}$', 'interpreter','latex', 'FontSize', 16);
str = sprintf(", N=%d, B=%d",N,BB(2));
title('Total Agent Error'+str,'Interpreter','latex', 'FontSize', 16);
grid on;
% lgd = legend('$B=5$', '$B=25$', '$B=50$');
% lgd.Interpreter = 'latex';
% lgd.FontSize = 14;
% lgd.Location = 'northeast';

figure(2)
semilogy(t,alpha(:,1),'b')
hold on;
semilogy(t,beta(:,1),'Color','r');
hold on;
semilogy(t,delta(:,1),'Color','g');
ylim([1e-7 1e5])
xlabel('Iterations $(k)$','interpreter','latex', 'FontSize', 16);
str = sprintf(", N=%d, B=%d",N,BB(1));
title({'Convergence of $\alpha \big( k;t_\ell \big)$ and $\beta \big( k;t_\ell \big)$'+str},'Interpreter','latex', 'FontSize', 16)
grid on;
%legend({'$\alpha \big( k;t_\ell \big)$','$\beta \big( k;t_\ell \big)$'},'Interpreter','latex', 'FontSize', 16)
lgd = legend('$\alpha \big( k;t_\ell \big)$','$\beta \big( k;t_\ell \big)$','$\delta \big( k;t_\ell \big)$');
lgd.Interpreter = 'latex';
lgd.FontSize = 16;
lgd.Location = 'northeast';

figure(3)
semilogy(t,alpha(:,1),'b')
hold on;
semilogy(t,alpha(:,2),'r')
hold on
semilogy(t,alpha(:,3),'g')
hold on
xlabel('Iterations $(k)$','interpreter','latex', 'FontSize', 16);
ylabel('$J(x(k),y(k);t_\ell) - J(x^*(t_\ell), y^*(t_\ell);t_\ell)$', 'Interpreter','latex', 'FontSize', 16);
str = sprintf(", N=%d",N);
title({'Convergence of $\alpha \big( k;t_\ell \big)$'+str},'Interpreter','latex', 'FontSize', 16)
grid on;
lgd = legend('$B=5$', '$B=25$', '$B=50$');
lgd.Interpreter = 'latex';
lgd.FontSize = 14;
lgd.Location = 'northeast';

% figure(1)
% plot(t,u_error);
% hold on;
% plot(t,y_error);
% xlabel('Iterations (k)','interpreter','latex');
% % ylabel('$\| \big( x^{i}(k), y^{i}(k) \big)- \big( x^*(t_\ell), y^*(t_\ell) \big) \|_{2}$', 'interpreter','latex','FontSize', 16);
% ylim([0, max(max(agent_error))+10]);
% title('Agent Error (N=6)');
% grid on;
% lgd = legend('$\sum^{N}_{i=1} \| x^{i}(k)- x^*(t_\ell) \|_{2}$','$\sum^{N}_{i=1} \|  y^{i}(k))- y^*(t_\ell))\|_{2}$');
% lgd.Interpreter = 'latex';
% lgd.FontSize = 14;
% lgd.Location = 'northeast';

% figure(3)
% plot(t,error);
% xlabel('Iterations');
% ylabel('$\sum^{N}_{i=1} \| (x^{i}(k), y^{i}(k))-(x^*(t_\ell), y^*(t_\ell))\|_{2}$', 'interpreter','latex', 'FontSize', 16);
% title('Total Error (N=6)')
% grid on;

% figure(4)
% semilogy(t,alpha,'b')
% hold on;
% semilogy(t,beta,'Color','r');
% hold on;
% semilogy(t,delta,'Color','g');
% xlabel('Iterations');
% % ylabel('$J(x(k),y(k);t_\ell) - J(x^*(t_\ell), y^*(t_\ell);t_\ell)$', 'Interpreter','latex', 'FontSize', 16);
% str = sprintf(", N=%d, B=%d",N,B);
% title({'Convergence of $\alpha \big( k;t_\ell \big)$ and $\beta \big( k;t_\ell \big)$'+str},'Interpreter','latex', 'FontSize', 16)
% grid on;
% %legend({'$\alpha \big( k;t_\ell \big)$','$\beta \big( k;t_\ell \big)$'},'Interpreter','latex', 'FontSize', 16)
% lgd = legend('$\alpha \big( k;t_\ell \big)$','$\beta \big( k;t_\ell \big)$','$\delta \big( k;t_\ell \big)$');
% lgd.Interpreter = 'latex';
% lgd.FontSize = 14;
% lgd.Location = 'southeast';

% figure(5)
% semilogy(t,beta,'Color','b');
% hold on;
% semilogy(t,delta,'Color','r');
% hold on;
% semilogy(t,deltaUB,'Color','g');
% grid on;
% xlabel('iterations')
% legend({'$\beta_{t} (k)$','$\delta (k)$','$B_{t}m \Vert C \Vert^{2} \beta_{t} (k)$'},'Interpreter','latex', 'FontSize', 12)
% title({'Lemma 5: $\delta (k) \leq B_{t}m \Vert C \Vert^{2} \beta_{t} (k)$'},'Interpreter','latex', 'FontSize', 16)



%%
%         dfdu = (Q(:,:,i) + C'*P(:,:,i)*C) * z(1:sum(n),:) + q(:,i) + C'*p(:,i); % gradient formula
%         u_new(:,:) = z(1:sum(n),:) - gamma * dfdu; % gradient step
%         u_new(find(u_new<0)) = 0; % projection
        
%         dfdz = R(:,:,i) * z + r(:,i); % gradient formula
%         dfdz = [Q(:,:,i) * z(1:sum(n),:) + r(1:sum(n),i); C * (P(:,:,i) * z(sum(n)+1:end,:) + r(sum(n)+1:end,i))]; % gradient formula
        %dfdu = r(1:sum(n),i) + C'*r(sum(n)+1:end,:) + (Q(:,:,i) + C'*P(:,:,i)*C) *  z(1:sum(n),:); % gradient formula
