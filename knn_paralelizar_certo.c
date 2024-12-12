#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <windows.h>

#include <omp.h>

double obterTempoEmSegundos() {
    LARGE_INTEGER frequency, start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return (double)start.QuadPart / frequency.QuadPart;
}

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

    #pragma omp parallel for num_threads(8)
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
    #pragma omp parallel for num_threads(8)
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
double * criar_YTest(double **matriz_treino, int linhas_treino, double **matriz_teste, int linhas_teste, double *y_treino, int largura, int k) {
    double *y_teste = (double *)malloc(linhas_teste * sizeof(double));

    #pragma omp parallel for num_threads(8)
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
    return y_teste;
}

// Função para calcular o erro absoluto médio
/*double calcular_erro_absoluto_medio(double *xTest, double *y_previsto, int tamanhoXTest, int largura) {
    if (tamanhoXTest == 0) {
        fprintf(stderr, "Erro: Divisão por zero em calcular_erro_absoluto_medio. O tamanho é zero.\n");
        return NAN; // Retorna NaN explicitamente para indicar erro.
    }
    double soma = 0.0;

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < tamanhoXTest - largura ; i++) {
        soma += fabs(xTest[i + largura] - y_previsto[i]);
    }

    return soma / tamanhoXTest;
}*/

double calcular_erro_absoluto_medio(double *xTest, double *y_previsto, int tamanhoXTest, int largura, int altura) {
    double soma = 0.0;

for (int i = 0; i < tamanhoXTest; i++) {
    soma += fabs(xTest[i + largura + altura] - y_previsto[i]);
}

    return soma / tamanhoXTest;
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
char* nomeArquivoY(const char* nomeConjunto){
    char *ptr = NULL;
    char *nomeArquivoY = strdup(nomeConjunto); // Copiar a string original
    if ((ptr = strstr(nomeArquivoY, "xtest"))) memcpy(ptr, "yTest", 5);
    return nomeArquivoY;
}
// Função principal do algoritmo KNN
void knn(const char* nomeConjunto, double **matrizTrain, int linhasTrain, double *yTrain, int tamanhoTrain, double *xTest, int tamanhoTest, int largura, int altura, int k) {
        
    // Variáveis para medição do tempo
    double inicio, fim;
    double tempo_total;

    // Início da contagem de tempo
    inicio = obterTempoEmSegundos();

    int linhasTest;
    double **matrizTest = criar_matriz(xTest, tamanhoTest, largura, altura, &linhasTest);
    double *yTest = criar_YTest(matrizTrain, linhasTrain, matrizTest, linhasTest, yTrain, largura, k);

    fim = obterTempoEmSegundos();
    tempo_total = ((double)(fim - inicio));
    printf("Tempo total de execução do teste %s: %.20f segundos\n", nomeConjunto, tempo_total);
    
    salvar_YTest_em_arquivo(nomeArquivoY(nomeConjunto), yTest, linhasTest);

    double erro = calcular_erro_absoluto_medio(xTest, yTest, linhasTest, largura, altura);
    
    printf("Erro absoluto médio do conjunto de dados %s: %f\n", nomeConjunto, erro);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < linhasTest; i++) free(matrizTest[i]);

    free(matrizTest);
    free(yTest);
}

double* criar_yTreino(double* xTrain, int linhasTrain, int largura, int altura){
    double *yTrain = (double *)malloc(linhasTrain * sizeof(double));

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < linhasTrain; i++) {
        yTrain[i] = xTrain[i + largura + altura - 1];
    }
    return yTrain;
}

// Função main para executar o código
int main() {
    double *xTrain, *xTest;
    int tamanhoTrain, tamanhoTest;
    int linhasTrain=0, largura = 3, altura = 1, k = 2;
    const char *nomesArquivosTest[] = {
        "dados_xtest_10.txt",
        "dados_xtest_30.txt",
        "dados_xtest_50.txt",
        "dados_xtest_100.txt",
        "dados_xtest_1000.txt",
        "dados_xtest_100000.txt",
        "dados_xtest_1000000.txt",
        "dados_xtest_10000000.txt"
        };

    int totalArquivos = sizeof(nomesArquivosTest) / sizeof(nomesArquivosTest[0]);
    // Leitura do conjunto de treino e a partir dele construção da matriz de treino e do y de treino
    ler_arquivo("dados_xtrain.txt", &xTrain, &tamanhoTrain);
    double **matrizTrain = criar_matriz(xTrain, tamanhoTrain, largura, altura, &linhasTrain);
    double *yTrain = criar_yTreino(xTrain, linhasTrain, largura, altura);
    
    for (int i = 0; i < totalArquivos; i++) {
        double *xTest = NULL; // Inicializa xTest para cada arquivo
        int tamanhoTest = 0;

        // Lê o arquivo atual
        ler_arquivo(nomesArquivosTest[i], &xTest, &tamanhoTest);

        knn(nomesArquivosTest[i], matrizTrain, linhasTrain, yTrain, tamanhoTrain, xTest, tamanhoTest, largura, altura, k);

        
        free(xTest); // Libera a memória alocada para xTest
    }

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < linhasTrain; i++) free(matrizTrain[i]);

    free(matrizTrain);
    free(yTrain);
    free(xTrain);

    return 0;
}
