import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm,Fsc):
    np.savetxt("cmatriz.csv", cm, fmt="%d")
    np.savetxt("fscores.csv", Fsc, fmt="%.10f")

def load_w():
    w = np.load('w_snn.npz')
    return {key: w[key] for key in w.keys()}



def load_data_test():
    X = np.loadtxt("dtestX.csv", delimiter=",")
    Y = np.loadtxt("dtestY.csv", delimiter=",")
    return X, Y
    

# Beginning ...
def main():			
    param = ut.load_cnf()
    xv,yv = load_data_test()
    W = load_w()
    a,_ = ut.forward(xv.T,W,param)   
    Y_predicted = a['a2'] if param['nOcultosCapa2'] == 0 else a['a3']   		
    cm,Fsc = ut.metricas(yv.T,Y_predicted) 	
    save_measure(cm,Fsc)
		

if __name__ == '__main__':   
	 main()

