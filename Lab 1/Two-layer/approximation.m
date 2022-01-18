clear all;
clc

%Data


errorThreshhold = 1e-3;
eta = 0.001;
criteria = 0;

rng(1);
x = [-5:0.5:5]';
y = [-5:0.5:5]';
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
ndata = length(x)*length(y);
figure(1)
mesh(x,y,z);
title("Gauss function")

%%
targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid(x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

epochsl = 1:100:1000;
ne = length(epochsl);
Nhidden = 8;
errorThreshhold = 1e-3;
eta = 0.01;

meanSquaredErrorE = NaN([5, ne]);

for i = 1:ne
    for j = 1:5
        network = Backprop(patterns, targets, epochsl(i), errorThreshhold, eta, 8, false);
        outputs = predict(network, patterns);
        meanSquaredErrorE(j,i) = mean((targets - outputs).^2);
    end
end

figure(1)
plot(epochsl, mean(meanSquaredErrorE))
xlabel("Epoch")
ylabel("MSE")
title("MSE of training data depending on number of epochs")
%%
epochs = 200;
networkG = Backprop(patterns, targets, epochs, errorThreshhold, eta, 8, true);
out = predict(networkG, patterns);
gridsize = length(x);
zz = reshape(out, gridsize, gridsize);


%%
n = length(patterns);
p = randperm(n);
X = patterns(:,p);
Y = targets(p);
Nhidden = [1 3 5 10 25];
%1

% 80 
X_train1 = X(:,1:0.8*n);
X_test1 = X(:,0.8*n+1:end);
Y_train1 = Y(1:0.8*n);
Y_test1 = Y(0.8*n+1:end);
nT = length(Y_test1);

%%
accuracy = NaN([10, 5]);
mse = NaN([10, 5]);
for i = 1:5
        for j = 1:10
            network = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, Nhidden(i), false);
            outputs = predict(network, X_test1);
            accuracy(j,i) = sum((Y_test1 == outputs))/length(Y_test1);
            mse(j,i) = mean((Y_test1 - outputs).^2);
        end
end
figure(2)
plot(Nhidden, mean(mse))
xlabel("Number of hidden nodes")
ylabel("MSE")
title("Mean square Error depending en number of neurons in hidden layer")

%%
network3 = Backprop(patterns, targets, epochs, errorThreshhold, eta, 3, true);
%%
network30 = Backprop(patterns, targets, epochs, errorThreshhold, eta, 30, true);

%%
%2
% 80 

networkb1 = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, 20, false);
outputsb1 = predict(networkb1, X_test1);
mse1 = NaN([10, 1]);
variance1 = NaN([10, 1]);
bias1 = NaN([10, 1]);
for i = 1:10
    networkb1 = Backprop(X_train1, Y_train1, epochs, errorThreshhold, eta, 20, false);
    outputsb1 = predict(networkb1, X_test1);
    variance1(i) = var(outputsb1)
    bias1(i) = abs(mean(outputsb1)-mean(Y_test1));
    mse1(i) = mean((Y_test1 - outputsb1).^2);
end


%%
%20
X_train2 = X(:,1:0.2*n);
X_test2 = X(:,0.2*n+1:end);
Y_train2 = Y(1:0.2*n);
Y_test2 = Y(0.2*n+1:end);


mse2 = NaN([10, 1]);
variance2 = NaN([10, 1]);
bias2 = NaN([10, 1]);
for i = 1:10
    networkb = Backprop(X_train2, Y_train2, epochs, errorThreshhold, eta, 20, false);
    outputsb = predict(networkb, X_test2);
    variance2(i) = var(outputsb);
    bias2(i) = abs(mean(outputsb)-mean(Y_test2));
    mse2(i) = mean((Y_test2 - outputsb).^2);
end

%%
compare = [mean(variance1) mean(bias1) mean(mse1);
           mean(variance2) mean(bias2) mean(mse2);]
       
       
       
       



%%
%Two-layer perceptron 

function result = phi(x)
    result = (2./(1+exp(-x)) - 1);
end

function result = phi_prime(x)
    result = (1+phi(x)).*(1-phi(x))./2;
end

function network = Backprop(x, y, iterations, errorThreshhold, eta, Nhidden, plt)
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
    err = Inf;
    dw = 0;
    dv = 0;
    
    %Training Process
    while (epoch < iterations) && (abs(err) > errorThreshhold)
    %for epoch = 1:iterations 
        alpha = 0.9;
        
        %Forward pass

        hin = w * [patterns ; ones(1, N)];
        hout = [phi(hin) ; ones(1, N)];
        
        oin = v * hout;
        out = phi(oin);
        
        if plt
            %Show evolution
            x = [-5:0.5:5]';
            y = [-5:0.5:5]';

            gridsize = length(x);
            zz = reshape(out, gridsize, gridsize);
            figure(6)
            mesh(x, y, zz);
            axis([-5 5 -5 5 -0.7 0.7]);
            drawnow;
        end
        
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

function [class] = classif(Y, criteria)
    class=[];
    for i=Y
        if i >= criteria 
            class = [class 1];
        else
            class = [class -1];
        end
    end
end



