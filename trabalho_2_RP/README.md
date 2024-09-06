# Trabalho 2: Classificação de Movimentos de Mão usando k-NN

## Descrição

Neste trabalho, o objetivo é implementar o classificador **k-NN** (k-vizinhos mais próximos) para classificar dois tipos de movimentos de mão ("abrir a mão" e "mão para baixo") com base em dados de acelerômetros obtidos de uma luva sensorial. Foram utilizados diferentes valores de k, e o desempenho foi avaliado utilizando **validação cruzada K-fold** (K=10). Além disso, foram extraídos diversos atributos dos sinais para auxiliar na classificação.

### Arquivos de entrada:
- **InputData.mat**: Matriz de tamanho 1500x120, onde cada coluna representa uma amostra de movimento.
- **OutputData.mat**: Vetor de tamanho 120, com as classes de saída (-1 e +1) correspondentes aos movimentos.

## Como Executar
1. Certifique-se de que os arquivos `InputData.mat` e `OutputData.mat` estão no mesmo diretório que o script MATLAB.
2. Execute o script `Trabalho2_KNN.m` para classificar os sinais e gerar os gráficos de dispersão.   

---
Para mais detalhes sobre a prática de simulação, consulte o arquivo PDF fornecido na pasta.