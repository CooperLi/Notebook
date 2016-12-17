%%% This code is used in the Perceptron simulation.
%%% Related files: initPerceptron.m, plotInput.m,
%%% trainPerceptron.m, plotDecisionSurf.m

clf  %%% clear the figure window
hold on  %%% 'hold on' to allow incremental plotting of input and weights
view(-37.5,30)
plotPerceptronInput
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Classification boundary for a 3-input, 1-output perceptron')
grid on
for patNum = 1:nPats,
  weights = trainPerceptron(patNum, weights,input,target,lRate);
end
plotDecisionSurf
weights
