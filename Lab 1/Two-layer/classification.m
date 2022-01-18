%%
clear all;
clc
%3.2.1
criteria = 0;

%Data generation


rng(1);

n = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
classA(1,:) = [ randn(1,round(0.5*n)) .* sigmaA - mA(1), ...
randn(1,round(0.5*n)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

patterns = [classA classB];
targets = [zeros(n, 1)-1, zeros(n, 1)+1];

rng(2);
p = randperm(2*n);
X = patterns(:,p);
Y = targets(p);

figure(1)
plot(classA(1,:), classA(2,:), "bo", classB(1,:), classB(2,:), "ro")
legend("classA", "classB")



%%

%1
epochsl = 1:100:1000;
ne = length(epochsl);
Nhidden = 8;
errorThreshhold = 1e-3;
eta = 0.01;

meanSquaredErrorE = NaN([5, ne]);

for i = 1:ne
    for j = 1:5
        network = Backprop(X, Y, epochsl(i), errorThreshhold, eta, Nhidden);
        outputs = predict(network, X);
        outputs = classify(outputs, criteria);
        meanSquaredErrorE(j,i) = mean((Y - outputs).^2);
    end
end


%%
epochs = 1000;
Nhidden = [3 8 15 20 30];
accuracy = NaN([5, 1]);
meanSquaredError = NaN([5, 1]);

for i = 1:5
        network = Backprop(X, Y, epochs, errorThreshhold, eta, Nhidden(i));
        outputs = predict(network, X);
        outputs = classify(outputs, criteria);
        accuracy(i) = sum((Y == outputs))/length(Y);
        meanSquaredError(i) = mean((Y - outputs).^2);
end


figure(1)
subplot(2, 1, 1);
plot(epochsl, mean(meanSquaredErrorE))
xlabel("Epoch")
ylabel("MSE")
title("MSE of training data depending on number of epochs")

subplot(2, 1, 2);
plot(Nhidden, meanSquaredError)
xlabel("Number of hidden nodes")
ylabel("MSE")
title("MSE of training data depending on number of neurons in hidden layer")


%%
%Best hidden nodes
[best, indmax] = max(accuracy);
network = Backprop(X, Y, epochs, errorThreshhold, eta, Nhidden(indmax));
outputs = predict(network, X);
outputs = classify(outputs, criteria);

figure(3)
hold on
plot(classA(1,:), classA(2,:), "bo", classB(1,:), classB(2,:), "ro")
legend("classA", "classB")
title("Classification of training data with " + Nhidden(indmax) +" hidden nodes")

gscatter(X(1,:), X(2,:), outputs, 'br', 'xx')

hold off


%%

%2

% 80 vs 20
X_train1 = X(:,1:0.8*2*n);
X_test1 = X(:,0.8*2*n+1:end);
Y_train1 = Y(1:0.8*2*n);
Y_test1 = Y(0.8*2*n+1:end);

[accuracy1,  meanSquaredError1] =  evaluate(X_train1, Y_train1, X_test1, Y_test1, epochs, errorThreshhold, eta, Nhidden);


% 20 vs 80
X_train2 = X(:,1:0.2*2*n);
X_test2 = X(:,0.2*2*n+1:end);
Y_train2 = Y(1:0.2*2*n);
Y_test2 = Y(0.2*2*n+1:end);

[accuracy2,  meanSquaredError2] =  evaluate(X_train2, Y_train2, X_test2, Y_test2, epochs, errorThreshhold, eta, Nhidden);

figure(4)
subplot(2, 1, 1);
hold on
plot(Nhidden, accuracy1, 'b', Nhidden, accuracy2, 'r')
legend("80% Train vs 20% Test", "20% Train vs 80% Test", "Location", "southeast")
xlabel("Number of hidden nodes")
ylabel("Accuracy")
title("Accuracy depending en number of neurons in hidden layer")
hold off
subplot(2, 1, 2);
hold on
plot(Nhidden, meanSquaredError1, 'b', Nhidden, meanSquaredError2, 'r')
legend("80% Train vs 20% Test", "20% Train vs 80% Test")
xlabel("Number of hidden nodes")
ylabel("MSE")
title("Mean Square Error depending en number of neurons in hidden layer")
hold off

%%

% epoch vs sequential
hidden = 20;

networkEp = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, hidden);
outputsEp = predict(networkEp, X_test1);
outputsEp = classify(outputsEp, criteria);
accuracyEp = sum(Y_test1 == outputsEp)/length(Y_test1);
meanSquaredErrorEp = mean((outputsEp - Y_test1).^2);

networkSeq = BackpropSeq(X_train1, Y_train1, epochs, errorThreshhold, eta, hidden);
outputsSeq = predict(networkSeq, X_test1);
outputsSeq = classify(outputsSeq, criteria);
accuracySeq = sum(Y_test1 == outputsSeq)/length(Y_test1);
meanSquaredErrorSeq = mean((outputsSeq - Y_test1).^2);

compare = [accuracyEp accuracySeq;
           meanSquaredErrorEp meanSquaredErrorSeq]

       %%
%Boundary
h = 0.01;

network3 = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, 3);
network30 = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, 30);

figure(5)
subplot(2,1,1)
hold on
boundary(network3, h)

plot(classA(1,:), classA(2,:), "bo", classB(1,:), classB(2,:), "ro")
legend("classA", "classB",'Location','southwest')
title("Decision boundary for network with 3 hidden nodes")

hold off

subplot(2,1,2)
hold on
boundary(network30, h)

plot(classA(1,:), classA(2,:), "bo", classB(1,:), classB(2,:), "ro")
legend("classA", "classB",'Location','southwest')
title("Decision boundary for network with 30 hidden nodes")
hold off






























%%
%Two-layer perceptron 

function result = phi(x)
    result = (2./(1+exp(-x)) - 1);
end

function result = phi_prime(x)
    result = (1+phi(x)).*(1-phi(x))./2;
end

function network = Backprop(x, y, iterations, errorThreshhold, eta, Nhidden)
% x : a matrix of N input instances by M features
% y : a matrix of N input Targets associated with x by L features
% iterations : a user parameter defines the maximum number of iterations
% errorThreshold : a user parameter specifies the maximum number of
% non-fatal errors that can occur
% eta : a user parameter determines the step size at each
% iteration
% Nhidden : a scalar defnies the the number of neurons in hidden layer
    %Initialize of parameters of training
    patterns = x;
    targets = y;
    
    N = size(patterns, 2);
    
    
    %Initialize Network attributes
    inArgc = size(patterns, 1);
    outArgc = size(targets, 1);
    
    
    %Initialise randomly the matrix of Weights w and v
    scale = 0.1;
    w = normrnd(0, 1, [inArgc, Nhidden]) * scale;
    w = [w ; ones(1, Nhidden)];
    w = w';
    v = normrnd(0, 1, [Nhidden, outArgc]) * scale;
    v = [v ; ones(1, outArgc)];
    v = v';
    
    %Initialize stopping conditions
    epoch = 0;
    err = Inf;
    dw = 0;
    dv = 0;
    iterations
    %Training Process
    while (epoch < iterations) && (abs(err) > errorThreshhold)
    %for epoch = 1:iterations 
        alpha = 0.9;
        
        %Forward pass

        hin = w * [patterns ; ones(1, N)];
        hout = [phi(hin) ; ones(1, N)];
        
        oin = v * hout;
        out = phi(oin);
        
        %Backward pass
        delta_o = (out - targets) .* phi_prime(targets);
        delta_h = (v' * delta_o) .* phi_prime(hout);
        delta_h = delta_h(1:Nhidden, :);
        
        %Updating
        dw = (dw .* alpha) - (delta_h * patterns') .* (1 - alpha);
        dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
        
        
        w(:,1:2) = w(:,1:2) + dw .* eta;
        v = v + dv .* eta;
        
        %End Backprop
        epoch = epoch + 1;
        err = mean((out - targets).^2);
        
    end
    
    network.structure = [inArgc Nhidden outArgc];
    network.w = w;
    network.v = v;
    network.epoch = epoch;
    network.error = err;

end

function [out] = predict(network, inputs)
    N = size(inputs, 2);
    w = network.w;
    v = network.v;
    hin = w * [inputs ; ones(1, N)];
    hout = [phi(hin) ; ones(1, N)];

    oin = v * hout;
    out = phi(oin);
end

function boundary(network, h)
    x_range = [-2:h:2];
    y_range = [-1:h:1];
    [X, Y] = meshgrid(x_range, y_range);
    
    w = network.w;
    v = network.v;
    N = length(x_range);
    Z = [];
    for i=y_range
        inputs = [x_range(:) (zeros(N,1)+i)];
        out = predict(network, inputs');
        Z = [Z; (out>=0)];
    end
    contourf(X, Y, Z)
end

function [accuracy,  meanSquaredError] = evaluate(X_train, Y_train, X_test, Y_test, epochs, errorThreshhold, eta, Nhidden)
    accuracy = NaN([5, 1]);
    meanSquaredError = NaN([5, 1]);
    for i = 1:5
            network = Backprop(X_train, Y_train, epochs, errorThreshhold, eta, Nhidden(i));
            outputs = predict(network, X_test);
            outputs = classify(outputs, 0);
            accuracy(i) = sum(Y_test == outputs)/length(Y_test);
            meanSquaredError(i) = mean((Y_test-outputs).^2);
    end
end



function network = BackpropSeq(x, y, iterations, errorThreshhold, eta, Nhidden)
% x : a matrix of N input instances by M features
% y : a matrix of N input Targets associated with x by L features
% iterations : a user parameter defines the maximum number of iterations
% errorThreshold : a user parameter specifies the maximum number of
% non-fatal errors that can occur
% eta : a user parameter determines the step size at each
% iteration
% Nhidden : a scalar defnies the the number of neurons in hidden layer
    %Initialize of parameters of training
    patterns = x;
    targets = y;
    
    N = size(patterns, 2);
    
    
    %Initialize Network attributes
    inArgc = size(patterns, 1);
    outArgc = size(targets, 1);
    
    
    %Initialise randomly the matrix of Weights w and v
    w = unifrnd(-1, 1, inArgc, Nhidden);
    w = [w ; ones(1, Nhidden)];
    w = w';
    v = unifrnd(-1, 1, Nhidden, outArgc);
    v = [v ; ones(1, outArgc)];
    v = v';
    
    %Initialize stopping conditions
    epoch = 0;
    error = Inf;
    
    %Training Process
    
    for i = 1:N
        sample = randperm(N, 1);
        input = patterns(:,sample);
        target = targets(:,sample);
        %for epoch=1:iterations
        while epoch < iterations && error > errorThreshhold
            dw = 0;
            dv = 0;
            alpha = 0.9;

            %Forward pass
            hin = w * [input ; 1];
            hout = [phi(hin) ; 1];

            oin = v * hout;
            out = phi(oin);

            %Backward pass
            delta_o = (out - target) .* phi_prime(target);
            delta_h = (v' * delta_o) .* phi_prime(hout);
            delta_h = delta_h(1:Nhidden, :);

            %Updating
            dw = (dw .* alpha) - (delta_h * input') .* (1 - alpha);
            dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);


            w(:,1:2) = w(:,1:2) + dw .* eta;
            v = v + dv .* eta;

            %End Backprop
            epoch = epoch + 1;
            error = mean((out - target).^2)

        end
    end
    
    network.structure = [inArgc Nhidden outArgc];
    network.w = w;
    network.v = v;
    network.epoch = epoch;
    network.error = error;

end

function [class] = classify(Y, criteria)
    class=[];
    for i=Y
        if i >= criteria 
            class = [class 1];
        else
            class = [class -1];
        end
    end
end













