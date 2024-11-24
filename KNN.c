#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int contarLinhasArquivo(const char* nomeArquivo) {

    FILE* arquivo = fopen(nomeArquivo, "r");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return -1;
    }

    int linhas = 0;
    double valor;
    
    while (fscanf(arquivo, "%lf", &valor) == 1) {  // Conta cada valor lido como uma linha
        linhas++;
    }

    fclose(arquivo);  // Fecha o arquivo após a contagem
    return linhas;
}   

// Função para ler os dados de um arquivo e preencher um array
void ler_arquivo(const char *nome_arquivo, double **valores, int *tamanho) {
    *tamanho = contarLinhasArquivo(nome_arquivo);  // Obtém a quantidade de linhas

    if (*tamanho <= 0) {
        *valores = NULL;
        return;
    }

    FILE* arquivo = fopen(nome_arquivo, "r"); // Abre o arquivo novamente para leitura dos valores
    
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        *valores = NULL;
        return;
    }

    *valores = (double*)malloc(sizeof(double)* *tamanho);  // Aloca o array com o tamanho exato

    for (int i = 0; i < *tamanho; i++) {
        fscanf(arquivo, "%lf", &(*valores)[i]);  // Lê cada valor do arquivo
    }

    fclose(arquivo);  // Fecha o arquivo após a leitura
}

// Função para criar uma matriz baseada no vetor de entrada
double **criar_matriz(double *dados, int tamanho, int largura, int altura, int *linhas_matriz) {
    *linhas_matriz = tamanho - largura - altura + 1;
    if (*linhas_matriz <= 0) {
        fprintf(stderr, "Dimensões inválidas para criação da matriz\n");
        exit(EXIT_FAILURE);
    }

    double **matriz = (double **)malloc(*linhas_matriz * sizeof(double *));
    for (int i = 0; i < *linhas_matriz; i++) {
        matriz[i] = (double *)malloc(largura * sizeof(double));
        for (int j = 0; j < largura; j++) {
            matriz[i][j] = dados[i + j];
        }
    }

    return matriz;
}

// Função para calcular o vetor de distâncias
void calcular_distancias(double **matriz_treino, int linhas_treino, int largura, double *linha_teste, int n, double *distancias) {
    for (int i = 0; i < n; i++) {
        double soma = 0.0;
        for (int j = 0; j < largura; j++) {
            double diff = matriz_treino[i][j] - linha_teste[j];
            soma += diff * diff;
        }
        distancias[i] = soma;
    }
}

// Função para encontrar os índices das k menores distâncias
void encontrar_k_menores(double *distancias, int n, int k, int *indices) {
    for (int i = 0; i < k; i++) {
        double menor = INFINITY;
        int indice_menor = -1;
        for (int j = 0; j < n; j++) {
            if (distancias[j] < menor) {
                menor = distancias[j];
                indice_menor = j;
            }
        }
        indices[i] = indice_menor;
        distancias[indice_menor] = INFINITY; // Marcar como usado
    }
}

// Função para calcular o vetor YTest
void criar_YTest(double **matriz_treino, int linhas_treino, double **matriz_teste, int linhas_teste, double *y_treino, int largura, int k, double *y_teste) {
    for (int i = 0; i < linhas_teste; i++) {
        double *linha_teste = matriz_teste[i];
        double *distancias = (double *)malloc(linhas_treino * sizeof(double));

        calcular_distancias(matriz_treino, linhas_treino, largura, linha_teste, linhas_treino, distancias);

        int *indices = (int *)malloc(k * sizeof(int));
        encontrar_k_menores(distancias, linhas_treino, k, indices);

        double soma = 0.0;
        for (int j = 0; j < k; j++) {
            soma += y_treino[indices[j]];
        }
        y_teste[i] = soma / k;

        free(distancias);
        free(indices);
    }
}

// Função para calcular o erro absoluto médio
double calcular_erro_absoluto_medio(double *y_real, double *y_previsto, int tamanho) {
    double soma = 0.0;
    for (int i = 0; i < tamanho; i++) {
        soma += fabs(y_real[i] - y_previsto[i]);
    }
    return soma / tamanho;
}

void salvar_YTest_em_arquivo(const char *nome_arquivo, double *yTest, int tamanho) {
    FILE *arquivo = fopen(nome_arquivo, "w");
    if (!arquivo) {
        perror("Erro ao criar o arquivo");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tamanho; i++) {
        fprintf(arquivo, "%lf\n", yTest[i]);
    }

    fclose(arquivo);
    printf("Resultados de yTest salvos em '%s'.\n", nome_arquivo);
}

// Função principal do algoritmo KNN
void knn(double *xTrain, int tamanhoTrain, double *xTest, int tamanhoTest, int largura, int altura, int k) {
    int linhasTrain, linhasTest;
    double **matrizTrain = criar_matriz(xTrain, tamanhoTrain, largura, altura, &linhasTrain);
    double **matrizTest = criar_matriz(xTest, tamanhoTest, largura, altura, &linhasTest);

    double *yTrain = (double *)malloc(linhasTrain * sizeof(double));
    for (int i = 0; i < linhasTrain; i++) {
        yTrain[i] = xTrain[i + largura + altura - 1];
    }

    double *yTest = (double *)malloc(linhasTest * sizeof(double));
    criar_YTest(matrizTrain, linhasTrain, matrizTest, linhasTest, yTrain, largura, k, yTest);
    salvar_YTest_em_arquivo("yTest.txt", yTest, linhasTest);

    double erro = calcular_erro_absoluto_medio(yTrain, yTest, linhasTrain);

    printf("Erro absoluto médio: %f\n", erro);

    // Liberar memória
    for (int i = 0; i < linhasTrain; i++) free(matrizTrain[i]);
    for (int i = 0; i < linhasTest; i++) free(matrizTest[i]);
    free(matrizTrain);
    free(matrizTest);
    free(yTrain);
    free(yTest);
}

// Função main para executar o código
int main() {
    double *xTrain, *xTest;
    int tamanhoTrain, tamanhoTest;

    ler_arquivo("xTrain.txt", &xTrain, &tamanhoTrain);
    ler_arquivo("xTest.txt", &xTest, &tamanhoTest);
    int largura = 3, altura = 1, k = 2;
    knn(xTrain, tamanhoTrain, xTest, tamanhoTest, largura, altura, k);

    free(xTrain);
    free(xTest);
    scanf("%d", &largura);
    return 0;
}
