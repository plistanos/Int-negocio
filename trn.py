import pandas     as pd
import numpy      as np
import utility    as ut

# #Save weights and MSE  of the SNN
# def save_w_mse():
#     ...
#     return

# #gets Index for n-th miniBatch
# def get_Idx_n_Batch(n,x,N):
#     ...    
#     return(Idx)

# #miniBatch-SGDM's Training 
# def trn_minibatch(x,y,param):    
#     W,V   = iniWs()
#     ....
#     return(...)

#SNN's Training 
def train(x,y,param):  
    rows,columns = x.shape
    m = param['nClases']
    d = columns
    W,V   = ut.iniWs()
    MSE = []
    
    return(W,V)

# Load data to train the SNN
def load_data_trn():
    xe = pd.read_csv('dtrainX.csv',header=None)
    ye = pd.read_csv('dtrainY.csv',header=None)
    return(xe,ye)
    
   
# # Beginning ...
def main():
    param       = ut.load_cnf()            
    xe,ye       = load_data_trn()   
    # print(ye)
    W,Cost      = train(xe,ye,param)             
    # save_w_cost(W,Cost)
       
if __name__ == '__main__':   
	 main()

