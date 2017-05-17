function twoClassesClassificationSkeleton
    close all;
    %D = 2, dimensionality of the input (temp and humidity)
    maxValX1 = 15;
    maxValX2 = 15;
    %two features, generated randomly
    x1 = maxValX1 * rand(100, 1);
    x2 = maxValX2 * rand(100, 1);
    X = [x1 x2]; %X(1,:) = x_1, ... etc
    % one sample per row
    c1 = 1;
    c2 = 1;
    t_i = 1;
    for i  = 1 : size(X, 1)
        Y(i, 1) = realDiscriminant(X(i, :)); 
        %If the distance from the real discriminant function is more than
        %5, is taken as a valid sample
        %5 y 2 son las distancias entre muestras
        if(abs(Y(i, 1))>5)
            %a sample per row, of dimension D = 2
            %with dummy input
            Xd(t_i, :) = [1 X(i, :)];
            %the output of the discriminant surface with the sample as an
            %input its positive, then belongs to Class 1
            if(Y(i, 1)>0)            
                C1(c1, :) =  X(i, :);
                c1 = c1 + 1;
                T(t_i) = 0;
                t_i = t_i + 1;
            else           
                C2(c2, :) = X(i, :);
                c2 = c2 + 1;  
                T(t_i) = 1;
                t_i = t_i + 1;
            end
            
            
        end
    end
    %T is a column vector
    T = T';
    %plots the training samples
    figure; scatter(C1(:,1), C1(:, 2), 'x');
    hold on;
    scatter(C2(:,1), C2(:, 2), 'O');
    %Calculates the least squares weight array
    W = getW_leastSquares(Xd, T);
    %test 1
    for i = 1:size(C1, 1)
        yResC1(i) = getY(W, [1 C1(i, :)]);
    end
    
    for i = 1:size(C2, 1)
        yResC2(i) = getY(W, [1 C2(i, :)]);
    end
    %prueba con fisher
    C1n = [ones(size(C1, 1), 1) C1];
    C2n = [ones(size(C2, 1), 1) C2];
    l2 = (-W(1)-W(2)*Xd(:,2)/W(3));
    plot(Xd(:,2), l2);
    hold on;
    %fisher evaluation
    Wfish = fisherDA(C1n, C2n);
    for i = 1:size(C1, 1)
        yResC1Fish(i) = getYFish(Wfish', C1n(i, :)');
    end
    for i = 1:size(C2, 1)
        yResC2Fish(i) = getYFish(Wfish', C2n(i, :)');
    end
    
    figure; 
    scatter(yResC1Fish, yResC1Fish, 'x');
    hold on;
    scatter(yResC2Fish, yResC2Fish, 'yellow');
    
    y = -Wfish(1);
    scatter(y, y, 'p','filled', 'magenta');
    hold on;
    
    figure;
    histogram(yResC1Fish);
    hold on;
    histogram(yResC2Fish);
    
    %prueba con perceptron
    %perceptron needs T to be -1 or 1, not 0 or 1, needs to be corrected
    numIter = 1000;
    Wperc = perceptronTraining(C1n, C2n, numIter);
    for i = 1:size(C1, 1)
        yResC1Perc(i) = perceptronActivationFunc(Wperc, C1n(i, :));
    end
    for i = 1:size(C2, 1)
        yResC2Perc(i) = perceptronActivationFunc(Wperc, C2n(i, :));
    end
    
end

%Implements the perceptron training algorithm
function W = perceptronTraining(C1n, C2n, numIter)
    
end
%activation function of the perceptron algorithm
function f = perceptronActivationFunc(W, x) 
   
end

%C1 and C2 tagged data
%Implements the Fisher discriminant analysis
function w = fisherDA(C1, C2)
    [ic1,jc1] = size(C1);
    [ic2,jc2] = size(C2);
    MiuC1 = sum(C1)./ic1;
    MiuC2 = sum(C2)./ic2;
    Distancia = (MiuC1-MiuC2);
    SB = Distancia*Distancia';
    S1 = restaMat(MiuC1, C1);
    S1T = S1';
    S1M = [];
    for i = 1:ic1
        S1M = [S1M S1(i,:)*S1T(:,i)];
    end
    S1F = sum(S1M);
    S2 = restaMat(MiuC2, C2);
    S2T = S2';
    S2M=[];
    for i = 1:ic2
        S2M = [S2M S2(i,:)*S2T(:,i)];
    end
    S2F = sum(S2M);
    SW = S2F + S1F;
    SWInv = pinv(SW);
    w = SWInv * (MiuC1-MiuC2);
end

function resultado = restaMat(MiuC, C)
    resultado = [C(:,1) - MiuC(1) C(:,2)- MiuC(2) C(:,3)- MiuC(3)];
end

%Projects the given sample using the weight vector W
function y = getYFish(W, x)
 y = W'*x;
end

%Evaluates the min squares classification algorithm
function y = getY(W, x)
    y = W' * x';
 
    %w0 is the threshold used for decision
    if(W(1)<0)
        y = y - W(1)<0 ;
    else
        y = y - W(1)>0 ;
    end
end

%Gets the optimum hiperplane W with least squares
function W = getW_leastSquares(X, T)
    W=pinv(X'*X)*X'*T;
end

%function to approximate
function y = realDiscriminant(x)
    %y = 0 in x(1) = 0.5 and x(2)=0.5
    %w = [2 -2]
    y = 2 * x(1) - 2*x(2) - 2;
end