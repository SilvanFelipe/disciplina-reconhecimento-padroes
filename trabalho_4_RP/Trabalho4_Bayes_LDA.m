% Universidade Federal do Ceará - UFC
% Reconhecimento de Padrões - 2024.1
% Francisco Silvan Felipe do Carmo - 496641

% Implementação e avaliação do Classificador Bayesiano e LDA com Dados Contínuos

% Comando para ver o tempo de execução do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Carregar o arquivo .mat
load('Input1.mat'); 
load('Input2.mat'); 

X1 = Input1';
X2 = Input2';
y = [ones(2000, 1); 2*ones(2000, 1)];


% Gráfico de dispersão 2D dos dados (Input2)
figure;
gscatter(X1(:,1), X1(:,2), y, 'rb', '..', 12);
title('Gráfico de Dispersão 2D dos Dados da Base Input1');
xlabel('Atributo 1');
ylabel('Atributo 2');
legend('Classe 1', 'Classe 2');
grid off;

% Gráfico de dispersão 2D dos dados (Input2)
figure;
gscatter(X2(:,1), X2(:,2), y, 'rb', '..', 12);
title('Gráfico de Dispersão 2D dos Dados da Base Input2');
xlabel('Atributo 1');
ylabel('Atributo 2');
legend('Classe 1', 'Classe 2');
grid off;

% Definir K para K-fold
K = 10;

% Classificação e avaliação para a Base de Dados 1
[mean_acc_bayes1, std_acc_bayes1, mean_acc_lda1, std_acc_lda1] = kfold_classification(X1, y, K);

% Exibir resultados para a Base de Dados 1
fprintf('Resultados para a Base de Dados 1:\n');
fprintf('Acurácia do Classificador Bayesiano: %.2f%%\n', mean_acc_bayes1 * 100);
fprintf('Desvio Padrão do Classificador Bayesiano: %.4f\n', std_acc_bayes1);
fprintf('Acurácia do LDA: %.2f%%\n', mean_acc_lda1 * 100);
fprintf('Desvio Padrão do LDA: %.4f\n\n', std_acc_lda1);

% Classificação e avaliação para a Base de Dados 2
[mean_acc_bayes2, std_acc_bayes2, mean_acc_lda2, std_acc_lda2] = kfold_classification(X2, y, K);

% Exibir resultados para a Base de Dados 2
fprintf('Resultados para a Base de Dados 2:\n');
fprintf('Acurácia do Classificador Bayesiano: %.2f%%\n', mean_acc_bayes2 * 100);
fprintf('Desvio Padrão do Classificador Bayesiano: %.4f\n', std_acc_bayes2);
fprintf('Acurácia do LDA: %.2f%%\n', mean_acc_lda2 * 100);
fprintf('Desvio Padrão do LDA: %.4f\n', std_acc_lda2);

% Parando a verificação do tempo de execução
toc;

%% Explicar por que o desempenho foi melhor ou pior comparado ao da primeira base de dados
%  Resposta: O desempenho entre a segunda base e a primeira diferem, pois para a base com 
%  classes sobrepostas, o Classificador Bayesiano mantém um desempenho razoável, devido à 
%  sua capacidade de modelar distribuições probabilísticas, mas a acurácia é reduzida por
%  conta da sobreposição das classes. 
%  O LDA, por sua vez, apresenta desempenho muito pior, pois depende da separabilidade 
%  linear e da diferença entre as médias. Com médias iguais e covariâncias distintas, 
%  o LDA perde sua eficácia, resultando em desempenho inferior.
%% ----------------------------------------------------------------------------------------

% -------------------------------- FUNÇÕES --------------------------------

% Função para classificação Bayesiana
function y_pred_bayes = bayesian_classifier(X_train, y_train, X_test)
    classes = unique(y_train);
    priori = zeros(1, length(classes));
    posteriori = zeros(size(X_test, 1), length(classes));

    for i = 1:length(classes)
        % Encontrar as amostras da classe i e calcular a priori
        idx_samples = find(y_train == classes(i));
        priori(i) = length(idx_samples)/length(X_train);
        samples = X_train(idx_samples, :);

        % Calcular a média e a covariância da classe i
        mu = mean(samples, 1);
        cv = cov(samples);

        % Calcular a probabilidade posteriori para cada amostra de teste
        for j = 1:length(X_test)
            x = X_test(j, :);
            likelihood = gaussian_likelihood(x, mu, cv); 
            posteriori(j, i) = likelihood * priori(i);
        end
    end
    % Classificar com base na máxima probabilidade a posteriori
    [~, y_pred_idx] = max(posteriori, [], 2);
    y_pred_bayes = classes(y_pred_idx);
end

% Função para calcular a likelihood(verossimilhança) gaussiana
function likelihood = gaussian_likelihood(x, mu, cov_matrix)
    % x: Vetor de amostras de teste
    % mu: Média da distribuição
    % cov_matrix: Matriz de covariância

    % Dimensão do vetor de entrada
    N = length(x); 
    
    % Termo constante na frente do exponencial
    const_term = 1 / ((2 * pi)^(N/2) * sqrt(det(cov_matrix)));
    
    % Diferença entre o vetor de amostras e a média
    diff = x - mu;
    
    % Termo exponencial
    exp_term = exp(-0.5 * (diff * (cov_matrix \ diff')));
    
    % Likelihood
    likelihood = const_term * exp_term;
end

% Função para classificação usando LDA linear
function y_pred_lda = LDA_linear_classifier(X_train, y_train, X_test)
    % Calcular médias de cada classe
    mu1 = mean(X_train(y_train == 1, :), 1);
    mu2 = mean(X_train(y_train == 2, :), 1);

    % Calcular matrizes de covariância de cada classe
    S1 = cov(X_train(y_train == 1, :));
    S2 = cov(X_train(y_train == 2, :));

    % Matriz de dispersão within-class
    Sw = S1 + S2;

    % Calcular vetor de projeção w
    w = inv(Sw) * (mu1 - mu2)';

    % Normalizar w
    w = w / norm(w);

    % Projetar dados e teste em w
    z_test = X_test * w;

    % Definir limiar como ponto médio
    threshold = (mu1*w + mu2*w) / 2;

    % Classificar dados de teste
    y_pred_lda = zeros(length(z_test), 1);
    y_pred_lda(z_test >= threshold) = 1;
    y_pred_lda(z_test < threshold) = 2;

end

% Função para realizar o K-fold cross-validation e calcular acurácia
function [mean_acc_bayes, std_acc_bayes, mean_acc_lda, std_acc_lda] = kfold_classification(X, y, K)
    % Inicializar vetores para armazenar acurácias
    accuracy_bayes = zeros(K, 1);
    accuracy_lda = zeros(K, 1);
    
    % Definir os índices para K-fold
    indices = crossvalind('Kfold', y, K);

    % Loop K-fold
    for k = 1:K
        % Índices de teste e treino
        test_idx = (indices == k);
        train_idx = ~test_idx;
        
        % Dividir os dados em treino e teste
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);
        
        % Classificador Bayesiano 
        y_pred_bayes = bayesian_classifier(X_train, y_train, X_test);
    
        % Classificador Linear LDA Linear
        y_pred_lda = LDA_linear_classifier(X_train, y_train, X_test);
        
        % Calcular acurácia do LDA Linear para o fold atual
        accuracy_lda(k) = sum(y_pred_lda == y_test) / length(y_test);
    
        % Calcular a acurácia do Classificador Bayesiano para o fold atual
        accuracy_bayes(k) = sum(y_pred_bayes == y_test) / length(y_test);
    end

    % Calcular a média e o desvio padrão das acurácias dos classificadores
    mean_acc_bayes = mean(accuracy_bayes);
    std_acc_bayes = std(accuracy_bayes);

    mean_acc_lda = mean(accuracy_lda);
    std_acc_lda = std(accuracy_lda);
end
