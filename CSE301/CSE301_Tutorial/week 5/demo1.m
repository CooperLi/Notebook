
%=========================================================================
% Classification of an XOR problem with a multilayer perceptron
%=========================================================================
% PROBLEM DESCRIPTION:
% 4 clusters of data (A,B,C,D) are defined in a 2-dimensional input space.
% (A,C) and (B,D) clusters represent XOR classification problem.
% The task is to define a neural network for solving the XOR problem.

%=========================================================================
clear, close all, format compact, clc
% Step 1. 
% Define 4 clusters of input data
% number of samples of each class
K = 100;
% define 4 clusters of input data
q = .6; % offset of classes
A = [rand(1,K)-q; rand(1,K)+q];
B = [rand(1,K)+q; rand(1,K)+q];
C = [rand(1,K)+q; rand(1,K)-q];
D = [rand(1,K)-q; rand(1,K)-q];
% plot clusters
plot(A(1,:),A(2,:),'k+')
hold on
grid on
plot(B(1,:),B(2,:),'b*')
plot(C(1,:),C(2,:),'kx')
plot(D(1,:),D(2,:),'bd')
% text labels for clusters
text(.5-q,.5+2*q,'Class A')
text(.5+q,.5+2*q,'Class B')
text(.5+q,.5-2*q,'Class C')
text(.5-q,.5-2*q,'Class D')

% Define output coding for XOR problem
% encode clusters a and c as one class, and b and d as another class
a = -1; % a | b
c = -1; % -------
b = 1; % d | c
d = 1; %

% Prepare inputs & outputs for network training
% define inputs (combine samples from all four classes)
P = [A B C D];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B)) ...
repmat(c,1,length(C)) repmat(d,1,length(D)) ];

% Step 2
%Create and train a multilayer perceptron
% create a neural network
% net = feedforwardnet([5 3]); %for Matlab 2010 version only
nhidden = 3;
noutput = size(T,1);
net=newff(P,T,[nhidden,noutput],{'logsig','logsig'},'traingd');
% train net
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0;  % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
% train a neural network
[net,tr,Y,E] = train(net,P,T);
% show network
view(net)

%Step 3 
%plot targets and network response to see how good the network learns the data
figure;
plot(T','linewidth',2)
hold on
plot(Y','r--')
grid on
legend('Targets','Network response','location','best')
ylim([-1.25 1.25])

% Plot classification result for the complete input space
% generate a grid
span = -1:.01:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simulate neural network on a grid
aa=sim(net,pp);
%aa = full(aa);
% plot classification regions
figure(1)
mesh(P1,P2,reshape(aa,length(span),length(span))-5);
colormap cool
view(2)

