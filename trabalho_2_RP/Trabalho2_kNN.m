% Universidade Federal do Ceará - UFC
% Reconhecimento de Padrões - 2024.1
% Francisco Silvan Felipe do Carmo - 496641

% Implementação e avaliação do algoritmo K Nearest Neighbors (KNN) com validação cruzada

% Comando para ver o tempo de execução do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Configuração do ambiente e carregamento dos dados
load('InputData.mat');  % Carrega os dados de entrada
load('OutputData.mat');  % Carrega as classes de saída

X = InputData;  % Dados de entrada
y = OutputData;  % Classes de saída

% Definindo nomes dos atributos
attribute_names = {'Média', 'Desvio Padrão', 'Assimetria', 'Curtose', ...
                   'Valor Máximo', 'Valor Mínimo', 'Amplitude', 'Mediana', 'Valor Quadrático Médio', ...
                   'Entropia', 'Zero Crossing Rate', 'Energia', 'Coeficientes de Fourier'};

% Extraindo atributos de cada amostra
num_samples = size(X, 2);  % Número de amostras
num_attributes = length(attribute_names);  % Número de atributos
features = zeros(num_samples, num_attributes);  % Inicializa a matriz de atributos

% Loop para calcular os atributos de cada amostra
for i = 1:num_samples
    sample = X(:, i);  % Obtém a amostra

    % Calcula os atributos
    features(i, 1) = mean(sample);                % Média
    features(i, 2) = std(sample);                 % Desvio padrão
    features(i, 3) = skewness(sample);            % Assimetria
    features(i, 4) = kurtosis(sample);            % Curtose
    features(i, 5) = max(sample);                 % Valor máximo
    features(i, 6) = min(sample);                 % Valor mínimo
    features(i, 7) = range(sample);               % Amplitude
    features(i, 8) = median(sample);              % Mediana
    features(i, 9) = rms(sample);                 % Valor quadrático médio
    features(i, 10) = entropy(sample);            % Entropia
    features(i, 11) = zero_crossing_rate(sample); % Zero Crossing Rate
    features(i, 12) = sum(sample.^2);             % Energia
    % Coeficientes de Fourier (primeiros 5)
    fft_coeffs = fft(sample);
    features(i, 13:17) = abs(fft_coeffs(1:5))';
end

% Normalizando os atributos de forma que tenham média 0 e variância 1 (z-score)
normalized_features = normalize(features);

% Definindo valores de k para testar e número de folds para validação cruzada
k_values = [1, 2, 3, 5, 7, 9];  % Valores de k
num_folds = 10;  % Folds (K=10)
best_accuracy = 0;  % Inicializa a melhor acurácia
best_k = 0;  % Inicializa o melhor valor de k

% Loop para testar diferentes valores de k com validação cruzada
for k = k_values
    accuracy = k_fold_cross_validation(normalized_features, y, k, num_folds);
    fprintf('Acurácia média nos 10 folds com k=%d: %.2f%%\n', k, accuracy * 100);
    
    % Atualiza a melhor acurácia e o melhor valor de k
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_k = k;
    end
end

fprintf('Melhor acurácia média: %.2f%% com k=%d\n', best_accuracy * 100, best_k);

% Gerando gráficos de dispersão dos atributos 2D
figure_idx = 1;

% Loop para criar gráficos de dispersão para cada par de atributos
% Foi observado que uma acuracia de 99% com 13 atributos, mostrando que 
% sua relação com a maioria é boa

for i = 1:num_attributes
    for j = i+1:num_attributes
        if mod(figure_idx - 1, 9) == 0
            figure;  % Cria uma nova figura a cada 9 subplots para melhor visualização
        end
        subplot(3, 3, mod(figure_idx - 1, 9) + 1);  % Cria um subplot
        scatter(normalized_features(:, i), normalized_features(:, j), 15, y, 'filled');
        title(sprintf('%s vs %s', attribute_names{i}, attribute_names{j}));
        xlabel(attribute_names{i});
        ylabel(attribute_names{j});
        figure_idx = figure_idx + 1;
    end
end

% Parando a verificação do tempo de execução
toc;

% --------------------------- FUNÇÕES -------------------------------------

% Função para calcular a entropia
function e = entropy(signal)
    p = histcounts(signal, 'Normalization', 'probability');
    p(p == 0) = [];
    e = -sum(p .* log2(p));
end

% Função para calcular a taxa de cruzamento por zero
function zcr = zero_crossing_rate(signal)
    zcr = sum(abs(diff(signal > 0)));
end

% Função que calcula a distância euclidiana entre 2 vetores de atributos
function dist = euclidean_distance(x1, x2)
    dist = sqrt(sum((x1 - x2).^2));
end

% Função KNN
function predicted_label = kNN(X_train, y_train, X_test, k)
    num_test_samples = size(X_test, 1);  % Número de amostras de teste
    predicted_label = zeros(num_test_samples, 1);  % Inicializa o vetor de rótulos previstos

    % Loop para prever a classe de cada amostra de teste
    for i = 1:num_test_samples
        distances = zeros(size(X_train, 1), 1);  % Inicializa o vetor de distâncias

        % Loop para calcular a distância entre a amostra de teste e todas as amostras de treino
        for j = 1:size(X_train, 1)
            distances(j) = euclidean_distance(X_train(j, :), X_test(i, :));
        end

        % Ordena as distâncias e obtém os índices das amostras mais próximas
        [~, sorted_indices] = sort(distances);

        k_current = k;  % Inicializa o valor de k atual
        while k_current > 0
            nearest_neighbors = y_train(sorted_indices(1:k_current));  % Obtém os k vizinhos mais próximos

            % Conta as classes dos k vizinhos mais próximos
            class0 = sum(nearest_neighbors == -1);
            class1 = sum(nearest_neighbors == 1);

            % Se não houver empate, interrompe o loop
            if class0 ~= class1
                break;
            end

            % Reduz o valor de k em 1 e repete o processo
            k_current = k_current - 1;
        end

        % Atribui a classe mais frequente entre os k vizinhos à amostra de teste
        if class0 > class1
            predicted_label(i) = -1;
        else
            predicted_label(i) = 1;
        end
    end
end


% Função para validação cruzada k-fold
function accuracy = k_fold_cross_validation(X, y, k, num_folds)
    % Número de amostras
    num_samples = size(X, 1);

    % Embaralha os dados
    indices = randperm(num_samples);
    X = X(indices, :);
    y = y(indices);

    % Calcula o tamanho de cada fold
    fold_size = floor(num_samples / num_folds);
    accuracy = zeros(num_folds, 1);  % Inicializa o vetor de acurácias

    % Loop para realizar a validação cruzada k-fold
    for i = 1:num_folds
        % Índices para validação
        val_start = (i - 1) * fold_size + 1;
        val_end = i * fold_size;
        if i == num_folds
            val_end = num_samples;  % Garante que o último fold contenha todas as amostras restantes
        end

        % Conjunto de validação
        test_idx = false(num_samples, 1);
        test_idx(val_start:val_end) = true;

        % Conjunto de treinamento
        train_idx = ~test_idx;

        X_train = X(train_idx, :);  % Dados de treino
        y_train = y(train_idx);  % Rótulos de treino
        X_test = X(test_idx, :);  % Dados de teste
        y_test = y(test_idx);  % Rótulos de teste

        % Predição do modelo KNN
        predicted_label = kNN(X_train, y_train, X_test, k);

        % Calcula a acurácia
        accuracy(i) = sum(predicted_label == y_test) / length(y_test);
    end

    % Média da acurácia dos folds
    accuracy = mean(accuracy);
end
