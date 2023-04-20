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
    param['tminiBatch']      = int(diccionario[0][8])
    param['tasaAp']          = (diccionario[0][9])
    param['beta']            = (diccionario[0][10])
    param['max_iterations']  = int(diccionario[0][11])
    return(param)

def sort_data_random(X, Y, by_column=False):    
    rows, columns = X.shape
    if by_column:
      rnd_i = np.random.permutation(columns)
      rnd_X = X[:, rnd_i]
      rnd_Y = Y[:, rnd_i]
    else:
      rnd_i = np.random.permutation(rows)
      rnd_X = X[rnd_i]
      rnd_Y = Y[rnd_i]
    
    return rnd_X, rnd_Y


# Initialize weights for SNN-SGDM
def iniWs(x, param):
    A0_len = len(x)
    w = {'w1': iniW(param['nOcultosCapa1'], A0_len)}
    v = {'v1': np.zeros((param['nOcultosCapa1'], A0_len))}
    if param['nOcultosCapa2'] == 0:
        w['w2'] = iniW(param['nClases'], param['nOcultosCapa1'])
        v['v2'] = np.zeros((param['nClases'], param['nOcultosCapa1']))
    else:
        w['w2'] = iniW(param['nOcultosCapa2'], param['nOcultosCapa1'])
        v['v2'] = np.zeros((param['nOcultosCapa2'], param['nOcultosCapa1']))
        w['w3'] = iniW(param['nClases'], param['nOcultosCapa2'])
        v['v3'] = np.zeros((param['nClases'], param['nOcultosCapa2']))
    return w, v

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
def forward(x,w,param):
    z = {'z1': np.dot(w['w1'],x)}
    a = {'a1': act_function(z['z1'],param['actOculta'])}
    z['z2'] = np.dot(w['w2'],a['a1'])
    a['a2'] = act_function(z['z2'], param['actOculta'])
    if param['nOcultosCapa2'] == 0:
        return a,z
    z['z3'] = np.dot(w['w3'],a['a2'])
    a['a3'] = act_function(z['z3'],param['actOculta'])
    return a,z

#Activation function
def act_function(z,f_act):
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

#Feed-Backward of SNN
def gradW(a,z,w,xe,ye,param):
    layers = len(z)
    al = a[f'a{layers}'] 
    gW = {}
    for i in reversed(range(1, layers + 1)):
        if i == layers:
            eL = (1 / param['tminiBatch']) * (al - ye)
            delta = eL * derivate_act(z[f'z{i}'], 5)
            gW[f'gW{i}'] = np.dot(delta, a[f'a{i-1}'].T)
        else:
            delta = eL * derivate_act(z[f'z{i}'], param['actOculta'])
            gW[f'gW{i}'] = np.dot(delta, xe.T) if i == 1 else  np.dot(delta, a[f'a{i-1}'].T)
        eL = np.dot(w[f'w{i}'].T, delta)   
    cost = 1/(2 * param['tminiBatch']) * ((al - ye) ** 2)
    return gW, cost

# Update W and V
def updWV_sgdm(w,v,gW,param):
    layers = len(w)
    for i in range(1, layers + 1):
        v[f'v{i}'] = param['beta'] * v[f'v{i}'] + param['tasaAp'] * gW[f'gW{i}']
        w[f'w{i}'] = w[f'w{i}'] - v[f'v{i}']
    return w, v

# Measure
def metricas(Y,Y_predicted):
    cm = confusion_matrix(Y, Y_predicted)
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)
    f_score = 2 * ((precision * recall) / (precision + recall))
    f_score = np.nan_to_num(f_score, nan=0)
    return cm, f_score
    
#Confusion matrix
def confusion_matrix(Y,Y_predicted):
    cm = np.zeros((Y.shape[0], Y.shape[0]), dtype=int)
    for j in range(Y.shape[1]):
        max_i_real = np.argmax(Y[:, j])
        max_i_pred = np.argmax(Y_predicted[:, j])
        cm[max_i_real, max_i_pred] += 1
    return cm
#-----------------------------------------------------------------------
