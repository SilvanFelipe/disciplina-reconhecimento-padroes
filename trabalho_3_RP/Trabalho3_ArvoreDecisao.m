% Universidade Federal do Ceará - UFC
% Reconhecimento de Padrões - 2024.1
% Francisco Silvan Felipe do Carmo - 496641

% Implementação e avaliação do algoritmo de Árvore de Decisão com validação
% cruzada para classifcação de uma Base de Dados com Atributos Categóricos

% Comando para ver o tempo de execução do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Configuração do ambiente e carregamento dos dados
load('Dataset.mat');  % Carrega os dados de entrada

X = Dataset(:, 1:3);  % Dados de entrada
y = Dataset(:, 4);    % Classes de saída

% Executar a validação cruzada com K=10 e armazena as métricas de desempenho
[accuracy, sensitivity, specificity, precision, f1score] = cross_validation(X, y, 10);

% Exibir resultados
fprintf('Acurácia: %.2f%%\n', accuracy * 100);
fprintf('Sensibilidade: %.2f%%\n', sensitivity * 100);
fprintf('Especificidade: %.2f%%\n', specificity * 100);
fprintf('Precisão: %.2f%%\n', precision * 100);
fprintf('F1-Score: %.2f\n', f1score);

% Parando a verificação do tempo de execução
toc;

%---------------------------- FUNÇÕES ----------------------------

function tree = build_tree(X, y, atributos)
    % Critério de parada: se todas as amostras têm a mesma classe ou 
    % se não há mais atributos para dividir
    if length(unique(y)) == 1 || isempty(atributos)
        tree.leaf = true;
        tree.label = mode(y);  % Classe majoritária
        return;
    end

    % Escolha o melhor atributo para dividir os dados
    [bestAtributo, ~] = best_split(X, y, atributos);

    % Remova o atributo usado da lista de atributos disponíveis
    atributos = atributos(atributos ~= bestAtributo);

    % Crie o nó da árvore
    tree.leaf = false;
    tree.atributo = bestAtributo;
    values = unique(X(:, bestAtributo)); % Valores distintos do atributo selecionado
    
    % Para cada valor do atributo, cria subárvores recursivamente
    for i = 1:length(values)
        value = values(i);
        subX = X(X(:, bestAtributo) == value, :);
        subY = y(X(:, bestAtributo) == value);
        
        % Recursão para criar as subárvores
        tree.children{i} = build_tree(subX, subY, atributos);
        tree.children{i}.value = value;
    end
end

% Percorre todos os atributos disponíveis e calcula o ganho de informação para cada um, 
% selecionando o atributo que maximiza o ganho de informação.
function [bestAtributo, bestGain] = best_split(X, y, atributos)
    bestGain = -1;
    bestAtributo = -1;
    
    for i = 1:length(atributos)
        atributo = atributos(i);
        gain = information_gain(X, y, atributo);
        
        if gain > bestGain
            bestGain = gain;
            bestAtributo = atributo;
        end
    end
end

function entropy = calculate_entropy(y)

    % Verifica as k classes da base
    classes = unique(y);
    total_samples = length(y);

    % Calcula a entropia
    entropy = 0;
    for i = 1:length(classes)
        % Total de amostras referentes a cada classe
        objects = sum(y == classes(i));
        prob_priori = objects / total_samples; % probabilidade associada a cada classe
        entropy = entropy - prob_priori * log2(prob_priori);
    end
end

function gain = information_gain(X, y, atributo)

    % Entropia total
    H = calculate_entropy(y);

    values = unique(X(:, atributo));
    average_entropy = 0;

    % subbase é um vetor que contém os elementos de y 
    % que estão associados às linhas de X
    % Em seguida é calculado a entropia média das subbases
    for i = 1:length(values)
        subbase = y(X(:, atributo) == values(i));
        average_entropy = average_entropy + (length(subbase)/length(y)) * calculate_entropy(subbase);
    end
    % Ganho de informação
    gain = H - average_entropy;
end

% Função para fazer a predição de uma amostra utilizando a árvore de decisão
function label = predict(tree,sample)
    if tree.leaf
        label = tree.label; % Se for um nó folha, retorna a classe
    else
        value = sample(tree.atributo);  % Obtém o valor da amostra para o atributo
        for i = 1:length(tree.children)
            if tree.children{i}.value == value
                label = predict(tree.children{i}, sample);  % Recursão na subárvore correspondente
                return;
            end
        end
    end
end

% Função para realizar a validação cruzada K-fold
function [accuracy, sensitivity, specificity, precision, f1score] = cross_validation(X, y, K)

    indices = crossvalind('Kfold', y, K); % Divide os dados em K folds
    accuracies = zeros(1, K);
    sensitivities = zeros(1, K);
    specificities = zeros(1, K);
    precisions = zeros(1, K);
    f1scores = zeros(1, K);

    % Para cada fold, treina e testa o modelo
    for k = 1:K
        test_idx = (indices == k);
        train_idx = ~test_idx;

        % Conjunto de treino e teste
        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);

        % Atributos disponíveis para divisão
        atributos = 1:size(X, 2);
        tree = build_tree(X_train, y_train, atributos);

        % Predição para o conjunto de teste
        y_pred = zeros(size(y_test));
        for i = 1:length(y_test)
            y_pred(i) = predict(tree, X_test(i, :));
        end

        [accuracies(k), sensitivities(k), specificities(k), precisions(k), f1scores(k)] = calculate_metrics(y_test, y_pred);
    end

    % Calcula as médias das métricas ao longo dos K folds
    accuracy = mean(accuracies);
    sensitivity = mean(sensitivities);
    specificity = mean(specificities);
    precision = mean(precisions);
    f1score = mean(f1scores);
end

% Função para calcular as métricas de desempenho
function [accuracy, sensitivity, specificity, precision, f1score] = calculate_metrics(y_true, y_pred)
    TP = sum((y_true == 1) & (y_pred == 1)); % Verdadeiros Positivos (TP)
    TN = sum((y_true == 0) & (y_pred == 0)); % Verdadeiros Negativos (TN)
    FP = sum((y_true == 0) & (y_pred == 1)); % Falsos Positivos (FP)
    FN = sum((y_true == 1) & (y_pred == 0)); % Falsos Negativos (FN)

    accuracy = (TP + TN) / length(y_true);
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    precision = TP / (TP + FP);
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity);
end

