import pandas     as pd
import numpy      as np
import utility    as ut
import os

# Save Data from  Hankel's features
def save_data(X,Y):

    np.savetxt("dtrainX.csv", X['train'], delimiter=",")
    np.savetxt("dtrainY.csv", Y['train'], delimiter=",", fmt="%i")
    np.savetxt("dtestX.csv", X['test'], delimiter=",")
    np.savetxt("dtestY.csv", Y['test'], delimiter=",", fmt="%i")

# # normalize data 
def data_norm(data):
    a = 0.01
    b = 0.99
    for key in data.keys():
        df = pd.DataFrame(data[key])
        # Normalize all variables with max min method
        for label in df.columns:
            if (df[label].max() != df[label].min()):
                df[label] = (df[label]-df[label].min()) / \
                    (df[label].max()-df[label].min()) * (b-a) + a
            else:
                df[label] = a
        data[key] = df.values

    return data


# # Binary Label
def binary_labels(i,datF,param):
    rows,columns = datF.shape
    label = np.zeros((rows, param['nClases']))
    label[:,i] = 1
    
    return label

def elementosRango(x, lower_bound, upper_bound):
    cont = 0
    for num in x:
        if lower_bound <= num <= upper_bound:
            cont += 1

    return cont


# # Fourier spectral entropy
def entropy_spectral(X):

    amplitudes = np.abs(np.fft.fft(X))

    I_x = int(np.sqrt(len(amplitudes)))
    x_max = np.max(amplitudes)
    x_min = np.min(amplitudes)
    x_range = x_max - x_min

    E = 0
    prob = 0

    step_range = x_range / I_x
    lower_bound = x_min

    for _ in range(I_x):
        upper_bound = lower_bound + step_range
        prob = 0 if elementosRango(amplitudes, lower_bound, upper_bound) == 0 else elementosRango(
            amplitudes, lower_bound, upper_bound) / len(amplitudes)
        E += prob*np.log2(prob)

    return -E

# # Hankel-SVD
def hankel_svd(X,param,j):
    n = param['lFrame']
    ci = []
    for z in range(2**j):
        x=X[:,z]
        l = 2
        k= n-l+1
        h = np.zeros((l,k))
        for i in range(l):
            h[i,:] = x[i:i+k]
        U, s, V = np.linalg.svd(h)
        for j in range(l):
            c = []
            Uj = U[:,j].reshape(-1,1)
            Vj = V[j,:].reshape(1,-1)
            UV = np.dot(Uj,Vj)
            Sj = s[j]
            h = UV.dot(Sj)
            for i in range(k):
                c.append(h[0][i])
            for i in range(1,l):
                c.append(h[i][k-1])
            ci.append(c)
    ci = np.array(ci)
    rows,columns = ci.shape
    return(ci.reshape(columns,rows)) 

# # Hankel's features 
def hankel_features(X,param,columns):
    f = []
    for nFrame in range(param['nFrame']):
        x = X[(param['lFrame']*nFrame):(param['lFrame']+(param['lFrame']*nFrame))]
        x = x.reshape(-1,1)
        for j in range(param['nivelDesc']):
            x = hankel_svd(x,param,j)
        entropias = []
        valores_singulares = []
        rows,columns = x.shape
        for i in range(columns):  
            entropy_c = entropy_spectral(x[:,i])
            _, Svalues_C, _ = np.linalg.svd(x[:,i].reshape(1, -1), full_matrices=False)
            entropias.append(entropy_c)
            valores_singulares.append(Svalues_C[0])
        feature =  entropias + valores_singulares
        # print(np.array(feature).shape)
        f.append(feature)
    f=np.array(f)
    return(f) 


# # Obtain j-th variables of the i-th class
def data_class(x,j,i):
    variable = x[i][j]
    return(variable) 

def apilar_features(datF,f,j):
    rows,columns = f.shape
    for i in range(rows): 
        datF[rows + i] = f[i]
    return datF


def create_dtrn_dtst(X, Y, p):
    random_X, random_Y = ut.sort_data_random(X,Y)
   
    xe_ye = int(len(random_X) * p)

    data = {
        'train': random_X[:xe_ye, :],
        'test': random_X[xe_ye:, :],
    }

    labels = {
        'train': random_Y[:xe_ye, :],
        'test': random_Y[xe_ye:, :],
    }

    return data, labels


# # Create Features from Data
def create_features(H,param):
    
    Y = []
    X = []
    rows, columns = H[0].shape
    
    for i in range(param['nClases']):
        datF = np.zeros((0,2**param['nivelDesc']*2))
        for j in range(columns):
            x = data_class(H,j,i)
            f = hankel_features(x.to_numpy(),param,columns)
            datF = np.concatenate((datF,f))
        X.extend(datF)
        label = binary_labels(i,datF,param)
        Y.extend(label)
    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape)
    # print(Y.shape)
    X, Y = create_dtrn_dtst(X, Y, param['tasaAp'])
    return(X,Y) 


# # Load data from ClassXX.csv
def load_data(nombreData):
    ruta = './data/%s' %nombreData
    archivos = os.listdir(ruta)
    matrices = {}
    for indice,archivo in enumerate(archivos):
        if archivo.endswith('.csv'):
            ruta_archivo = os.path.join(ruta,archivo)
            df = pd.read_csv(ruta_archivo,header=None)
            matrices[indice]=df
    return(matrices) 

# # Parameters for pre-proc.
# def load_cnf():
#   ...
#   return 

# Beginning ...
def main():        
    param           = ut.load_cnf()	
    data            = load_data('Data' + str(param['nClases']))	
    X,Y             = create_features(data, param)
    X               = data_norm(X)
    
    save_data(X,Y)


if __name__ == '__main__':   
	 main()


