%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Plot the 3D planar surface which is orthogonal to the weight vector. %%%%
%%%% Inputs on this surface are at the border between turning the output %%%%
%%%% unit on and turning it off, because any input point on this surface %%%%
%%%% would result in a weighted summed input equal to zero.              %%%%
%%%% Points above this surface would result in a positive net input and  %%%%
%%%% would turn the output unit on. Points below would turn the unit off.%%%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% This code is used in the Perceptron simulation.
%%% Related files: initPerceptron.m, plotInput.m,
%%% trainPerceptron.m, runPerceptron.m

%%% Create a grid of (X1,X2,X3) values covering
%%% the region defined by
%%% weights(1)*X1 + weights(2)*X2 + weights(3)*X3 + weights(4) = 0

% Create vectors of co-ordinates for first 2 axes
x1=0:0.1:1;
x2=x1;

% Convert these to matrices of co-ordinates for plotting.
[X1,X2]=meshgrid(x1,x2);
% Create matrix of surface values at the 
% co-ordinates given in X1,X2. 
SURFACE=-1/weights(3)*(weights(4)*ones(size(X1))+weights(1)*X1+weights(2)*X2);
mesh(X1,X2,SURFACE); %% plot the grid points 

%% Plot the weights as a vector, passing through the point X1 = 0.5, X2 = 0.5
%% along the line in the direciton W1 X1 + W2 X2 + W3 X3 + W4 = 1
x1 = 0.5;
x2 = 0.5;
x3 = -1/weights(3)*(weights(4) + weights(1) * x1 + weights(2) * x2);
x3a = x3 + 1;
x3b = x3 - 1;
plot3([x1 x1],[x2 x2],[x3a x3b]);
