import pandas     as pd
import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(w,cost):
    np.savez("w_snn.npz", **w)
    np.savetxt("costo.csv", cost, fmt="%.10f")

# #gets Index for n-th miniBatch
def get_Idx_n_Batch(M, n):
    start = n * M
    end = start + M
    return np.arange(start, end)

#miniBatch-SGDM's Training 
def trn_minibatch(x,y,w,v,param):
    rows,columns = x.shape
    M = param['tminiBatch']
    B = int(columns/M)
    cost=[]
    for n in range(int(B)):
        Idx = get_Idx_n_Batch(M,n)
        xe = x[:,Idx]
        ye = y[:,Idx]
        a,z = ut.forward(xe, w, param)
        gW,cost = ut.gradW(a,z,w,xe,ye,param)
        W,V = ut.updWV_sgdm(w,v,gW,param)
    
    return(cost,W,V)

#SNN's Training 
def train(x,y,param):  
    rows,columns = x.shape
    m = param['nClases']
    MSE = []
    W,V   = ut.iniWs(x,param)
    for i in range(param['max_iterations']):
        X, Y = ut.sort_data_random(x,y,by_column=True)
        Cost,W,V = trn_minibatch(X,Y,W,V,param)
        MSE.append(np.mean(Cost))
        if i % 10 == 0:
            print(f'n Iterar-SGD: ',i,np.array(MSE)[i])
    return W,np.array(Cost)

# Load data to train the SNN
def load_data_trn():
    xe = pd.read_csv('dtrainX.csv',header=None)
    ye = pd.read_csv('dtrainY.csv',header=None)
    return(xe.to_numpy().T,ye.to_numpy().T)
    
   
# # Beginning ...
def main():
    param       = ut.load_cnf()            
    xe,ye       = load_data_trn()   
    # print(ye)
    W,Cost      = train(xe,ye,param)             
    save_w_mse(W,Cost)
       
if __name__ == '__main__':   
	 main()

