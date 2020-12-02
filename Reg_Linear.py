import numpy as np
import scipy
from itertools import chain
import matplotlib.pyplot as plt

#Regressao Logistica implementada com vetorizacao do Numpy. Ao inves de usar loops para iterar em vetores e matrizes, 
# as operacoes sao feitas em nivel delas. Para perfeito entendimento do codigo, recomenda-se revisao de operacoes de 
# multiplicacao de matrizes, transposicao de matrizes, produto escalar e um pouco de entendimento de vetorizacao em numpy


def plotter(eixoX, eixoY):
    fig, ax = plt.subplots()
    ax.plot(eixoX, eixoY)
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Custo')
    ax.set_title('Custo X Iterações')
    plt.show()

def sigmoide(z):
    """
    Computa a funcao logistica (sigmoide) de z

    Parametros:
    z -- Um escalar ou um vetor numpy de qualquer dimensão

    Retorno:
    s -- sigmoide(z)
    """

    s = 1 / (1+ np.exp(-z)) #np.exp(-z) implementa e^(-z)
    
    return s


def inicializa_com_zeros(dim):
    """
    Cria uma matriz w de dimensoes [dim][1], ou seja, um vetor em coluna, e o inicializa com zero. Inicializa b = 0.
    
    Parametros:
    dim -- o tamanho do vetor de coluna w (igual a quantidade de caracteristicas de x^(i))
    
    Retorno:
    w -- vetor coluna inicializado com dimensoes (dim, 1)
    b -- escalar inicializado com valor 0 (vies)
    """

    w = np.zeros((dim,1)) #cria um vetor de determinada dimensao e seta suas posicoes valendo 0
    b = 0

    assert(w.shape == (dim, 1)) #garante que o vetor tenha sido criado com as dimensoes adequadas
    assert(isinstance(b, float) or isinstance(b, int)) #garante que b eh um float ou int
    
    return w, b


def propaga(w, b, X, Y):
    """
    Realiza a propagacao pra frente e pra tras. Implementa a funcao de custo J(w,b) e seu gradiente.

    Parametros:
    w -- vetor de pesos de tamanho (n_x,1) 
    b -- vies, um escalar
    X -- conjunto de dados de dimensoes (n_x,m) -- instancias sao vistas como vetores-colunas
    Y -- um vetor contendo o ground truth (0 ou 1) de tamanho (1,m)

    Retorno:
    custo -- funcao de custo J(w,b) para regressao logistica
    dw -- gradiente da funcao de perda com respeito ao vetor w, portanto de mesmas dimensoes de w
    db -- gradiente da funcao de perda com respeito a b, portanto um escalar
    """
    
    m = X.shape[1] #X.shape[1] retorna o numero de instancias
    
    # Propagacao pra frente (Entra com instancias e pesos, sai com ativacao (y_hat) e custo)
    Y_hat = sigmoide(np.dot(w.T,X)+b)                                # computa todos os y_hat^(i) --> operacao vetorial, Y_hat tem dimensao (1,m), a mesma de Y
    custo = - 1/m * np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))  # computa funcao custo J(w,b) --> operacao vetorial, Y.shape == Y_hat.shape
    
    # Propagacao para tras (encontra os gradientes que irao posteriormente atualizar os pesos de w e b)
    dw = 1/m * np.dot(X,(Y_hat-Y).T)  #encontra o dw medio dentre os dw parciais (dw = 1/m * soma(x^(i)*(y_hat^(i) - y)))  --> operacao vetorial, dw tem dimensao (n_x,1)
    db = 1/m * np.sum(Y_hat-Y)        #encontra o db medio dentre os db parciais (db = 1/m * soma(y_hat^(i) - y))  --> operacao vetorial, dw tem dimensao (n_x,1)

    #garantias que dados estao no tipo/dimensao certos
    assert(dw.shape == w.shape) 
    assert(db.dtype == float)
    custo = np.squeeze(custo) #transforma vetor unitario em variavel
    assert(custo.shape == ())
    
    gradientes = {"dw": dw,
             "db": db}
    
    return gradientes, custo


def treina(w, b, X, Y, num_iteracoes, taxa_aprendizado, print_custo = False):
    """
    Otimiza w e b com base nos gradientes aprendidos, atraves do algoritmo de gradiente descendente
    
    Parametros:
    w -- vetor de pesos de tamanho (n_x,1) 
    b -- vies, um escalar
    X -- conjunto de dados de dimensoes (n_x,m) -- instancias sao vistas como vetores-colunas
    Y -- um vetor contendo o ground truth (0 ou 1) de tamanho (1,m)
    num_iteracoes -- numero de interacoes (epocas) para aprendizado do modelo
    taxa_aprendizado -- taxa de aprendizado a ser utilizada
    print_custo -- se verdade, imprimir custo a cada 10 epocas
    
    Retorno:
    parametros -- dicionario contendo os pesos w e b
    gradientes -- dicionario contendo os gradientes de w e b
    custos -- lista de todos os custos computados durante a treinamento. Utilize essa lista para plotar a curva de aprendizado
    """
    
    custos = []
    it = []
    
    for i in range(num_iteracoes):
        
        #Realiza uma passagem de propagacao pra frente e pra tras, obtem gradientes
        gradientes, custo = propaga(w, b, X, Y)
        dw = gradientes["dw"]
        db = gradientes["db"]
        
        #Atualiza w e b com base no gradiente
        w = w - taxa_aprendizado * dw       # --> operacao vetorial
        b = b - taxa_aprendizado * db           
        
        # Guarda os custos a cada 100 iteracoes (epocas)
        if i % 10 == 0:
            custos.append(custo)
        
        # Exibe o custo a cada 100 iteracoes, se print_custo == true
        if print_custo and i % 10 == 0:
            print ("Custo depois de interacao %i: %f" %(i, custo))
            it.append(i)
    
    parametros = {"w": w,
              "b": b}
    
    gradientes = {"dw": dw,
             "db": db}
    plotter(it, custos)
    return parametros, gradientes, custos


def prediz(w, b, X):
    '''
    Prediz se uma instancia eh 0 ou 1, utilizando os parametros w e b treinados atraves de regressao logistica
    
    Parametros:
    w -- vetor de pesos de tamanho (n_x,1) 
    b -- vies, um escalar
    X -- conjunto de dados de dimensoes (n_x,m) -- instancias sao vistas como vetores-colunas
    
    Retorno:
    Y_predicao -- um vetor numpy contendo as predicoes (0 ou 1) para os exemplos em X, de tamanho (1,m)
    '''
    
    m = X.shape[1]
    Y_predicao = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1) #forca que w tenha a dimensao de um vetor coluna de X    
    
    #Calcula Y_hat, as probabilidades das instancias de X pertencerem a classe 1
    Y_hat = sigmoide(np.dot(w.T,X)+b) # --> operacao vetorial, Y_hat tem dimensao (1,m)
    
    for i in range(Y_hat.shape[1]):
        
        # Converte probabilidades Y_hat para prediçoes 
        if (Y_hat[0,i] > 0.5):
            Y_predicao[0,i] = 1
        else:
            Y_predicao[0,i] = 0
    
    
    assert(Y_predicao.shape == (1, m))
    
    return Y_predicao


def constroi_modelo(X_treino, Y_treino, X_teste, Y_teste, num_iteracoes, taxa_aprendizado, print_custo = False):
    """
    Constroi o modelo de regressao logistica com base nas funcoes anteriores
    
    Parametros:
    X_treino -- conjunto de treinamento de dimensao (n_x, m_treino) que contem as instancias para treinamento
    Y_treino -- rotulos de treinamento de dimensao (1, m_treino)
    X_teste -- conjunto de teste de dimensao (n_x, m_teste) que contem as instancias para teste
    Y_teste -- rotulos de teste de dimensao (1, m_teste)
    num_iteracoes -- numero de interacoes (epocas) para aprendizado do modelo
    taxa_aprendizado -- taxa de aprendizado a ser utilizada
    print_custo -- se verdade, imprimir custo a cada 10 epocas
    
    Retorno:
    d -- dicionario contendo informacoes sobre o modelo
    """
       
    # inicializa parametros w e b com zeros
    w, b = inicializa_com_zeros(X_treino.shape[0]) 

    # treina o modelo, aprendendo e otimizando os parametros w e b
    parametros, gradientes, custos = treina(w, b, X_treino, Y_treino, num_iteracoes, taxa_aprendizado, print_custo)
    
    # Recupera do dicionario de parametros
    w = parametros["w"]
    b = parametros["b"]
    
    # Prediz o rótulo de treinamento e de teste    
    Y_predicao_treino = prediz(w, b, X_treino)
    Y_predicao_teste = prediz(w, b, X_teste)

    # Exibe a taxa de acuracia do treino e do teste
    # Taxa de acuracia = 100 - MAE*100
    # Erro medio absoluto (MAE) ~~> MAE = 1/m * soma(|y_predito^(i) - y^(i)|)
    print("Acuracia de treino: {} %".format(100 - np.mean(np.abs(Y_predicao_treino - Y_treino)) * 100))
    print("Acuracia de teste: {} %".format(100 - np.mean(np.abs(Y_predicao_teste - Y_teste)) * 100))

    
    d = {"custos": custos,
         "Y_predicao_teste": Y_predicao_teste, 
         "Y_predicao_treino" : Y_predicao_treino, 
         "w" : w, 
         "b" : b,
         "taxa_aprendizado" : taxa_aprendizado,
         "num_iteracoes": num_iteracoes}
    
    return d


def _indexing(x, indices):
    """
    :param x: array from which indices has to be fetched
    :param indices: indices to be fetched
    :return: sub-array from given array and indices
    """
    # np array indexing
    if hasattr(x, 'shape'):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]


def train_test_split(*arrays, test_size=0.30, shufffle=True, random_seed=1):
    """
    splits array into train and test data.
    :param arrays: arrays to split in train and test
    :param test_size: size of test set in range (0,1)
    :param shufffle: whether to shuffle arrays or not
    :param random_seed: random seed value
    :return: return 2*len(arrays) divided into train ans test
    """
    # checks
    assert 0 < test_size < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length * test_size))
    n_train = length - n_test

    if shufffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))


def main():
    file_data = open('tic-tac-toe.data', 'r')
    matrix = []
    for linha in file_data:
        linha = linha.replace("\n", "")
        linha = linha.replace("positive", "1")
        linha = linha.replace("negative", "0")
        linha = linha.replace("x", "2")
        linha = linha.replace("o", "3")
        linha = linha.replace("b", "4")
        linha = linha.split(',')
        matrix.append(linha)

    matrix = np.array(matrix)
    matrix = np.asarray(matrix, dtype=np.float64)

    Treino, Teste = train_test_split(matrix)
    Treino = np.transpose(Treino)
    Teste = np.transpose(Teste)
    X_treino = Treino[:-1, :]
    Y_treino = Treino[-1]
    X_teste = Teste[:-1, :]
    Y_teste = Teste[-1]

    num_iteracoes = int(input('Número de Iterações(50, 100, 1000): '))
    taxa_aprendizado = float(input('Taxa de Aprendizado(0.1, 0.01, 0.001): '))
    print_custo = int(input('(0 - False, 1 - True): '))

    constroi_modelo(X_treino, Y_treino, X_teste, Y_teste, num_iteracoes, taxa_aprendizado, print_custo)

    


main()