# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  
#load parameters to train the SNN
def load_cnf():
    # Leer el archivo excel cnf
    cnf = pd.read_csv('cnf.csv',header=None)
    #Convertir lo leido en un diccionario
    diccionario = cnf.to_dict()
    param = {}
    param['nClases']       = int(diccionario[0][0])
    param['nFrame']        = int(diccionario[0][1])
    param['lFrame']    = int(diccionario[0][2])
    param['nivelDesc']       = int(diccionario[0][3])
    param['nOcultosCapa1']   = int(diccionario[0][4])
    param['nOcultosCapa2']   = int(diccionario[0][5])
    param['actOculta']       = int(diccionario[0][6])
    param['prTraining']      = (diccionario[0][7])
    param['tminiBatch']      = (diccionario[0][8])
    param['tasaAp']          = (diccionario[0][9])
    param['beta']            = (diccionario[0][10])
    param['max_iterations']  = (diccionario[0][11])
    return(param)

def sort_data_random(X, Y, by_column=False):    
    if by_column:
      random_indexes = np.random.permutation(X.shape[1])
      random_X = X[:, random_indexes]
      random_Y = Y[:, random_indexes]
    else:
      random_indexes = np.random.permutation(X.shape[0])
      random_X = X[random_indexes]
      random_Y = Y[random_indexes]
    
    return random_X, random_Y

# Initialize weights for SNN-SGDM
def iniWs(Param):    
    ...
    return(W,V)

# Initialize weights for one-layer    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

def salida_function(z):
    d = (1.0) / (1.0 + np.exp(-z))
    return d

# # Feed-forward of SNN
def forward(x,w,v,f_act=5):
    # Capa oculta    
    z   = np.dot(w,x.T)
    h   = act_function(z,f_act)
    # Capa salida
    z2  = np.dot(v,h)
    d   = salida_function(z2)
    return h,d.T

#Activation function
def act_function(z,f_act=1):
    if   f_act == 1: # ReLu
        act_fun = np.maximum(0, z)
    elif f_act == 2: # L-ReLu
        act_fun = np.where(z>=0, z, 0.01*z)
    elif f_act == 3: # ELU
        alpha   = 1.6732
        act_fun = np.where(z>0 , z, alpha*(np.exp(z) - 1))
    elif f_act == 4: #SELU
        lam=1.0507; alpha=1.6732
        act_fun = np.where(z>0, z, alpha*(np.exp(z)-1)) * lam
    elif f_act == 5: #Sigmoid
        act_fun = (1.0) / (1.0 + np.exp(-z))
    return act_fun
# Derivatives of the activation funciton
def derivate_act(a,f_act=1):
    if   f_act == 1: # ReLu
        der_fun = (a > 0).astype(float)
    elif f_act == 2: # L-ReLu
        der_fun = np.where(a>=0, 1, 0.01)
    elif f_act == 3: # ELU
        alpha   = 1.6732
        der_fun = np.where(a>0, 1, alpha*np.exp(a))
    elif f_act == 4: #SELU
        lam=1.0507; alpha=1.6732
        der_fun = np.where(a>0, 1, alpha*np.exp(a)) * lam
    elif f_act == 5: #Sigmoid
        der_fun = (np.exp(-a)/((1+np.exp(-a))**2))
    return der_fun

# #Feed-Backward of SNN
# def gradW(...):    
#     ...    
#     return(...)    

# # Update W and V
# def updWV_sgdm(...):
#     ...    
#     return(...)

# Measure
def metricas(x,y):
    matriz_confusion = np.zeros(shape=(y.shape[1], x.shape[1]))
    # print("confussion_matrix: ",confussion_matrix)

    for real, prediccion in zip(y, x):
        matriz_confusion[np.argmax(real)][np.argmax(prediccion)] += 1    
    
    fscore = []
    for i, value in enumerate(matriz_confusion):
        TP = value[i]
        FP = matriz_confusion.sum(axis=0)[i] - TP
        FN = matriz_confusion.sum(axis=1)[i] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        fscore.append((2 * (precision * recall) / (precision + recall)))

    fscore.append(np.array(fscore).mean())
    pd.DataFrame(matriz_confusion).astype(int).to_csv('cmatriz.csv',index=False,header=False)
    pd.DataFrame(fscore).to_csv('fscores.csv',index=False,header=False)
    return (fscore)
    
# #Confusion matrix
# def confusion_matrix(z,y):
#     ...    
#     return(cm)
#-----------------------------------------------------------------------
