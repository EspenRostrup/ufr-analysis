"""Implementation is mostly copied from https://github.com/yifan0sun/DynamicWord2Vec 
"""

import numpy as np
import utils
import pickle as pickle
import os

def print_params(r,lam,tau,gam,emph,ITERS):
    
    print('rank = {}'.format(r))
    print('frob  regularizer = {}'.format(lam))
    print('time  regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param   = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))
    
if __name__=='__main__':
    import sys
    # PARAMETERS
    nw = 22597 # number of words in vocab
    aggregation_interval = 5
    T = range(1867,2022,aggregation_interval) # total number of time points
    cuda = True
    ITERS = 10 # total passes over the data
    lam = 10 #frob regularizer
    gam = 100 # forcing regularizer
    tau = 50  # smoothing regularizer
    r   = 200  # rank
    b = nw # batch size
    emph = 1 # emphasize the nonzero

    trainhead = f'dw2v/PPMI/{aggregation_interval}_year_interval/wordPairPMI_' # location of training data
    savehead = f'trained_models/dw2v/{aggregation_interval}_year_interval/emb_{r}/'
    if not os.path.exists(savehead):
        os.makedirs(savehead)

    foo = sys.argv
    for i in range(1,len(foo)):
        if foo[i]=='-r':    r = int(float(foo[i+1]))        
        if foo[i]=='-iters': ITERS = int(float(foo[i+1]))            
        if foo[i]=='-lam':    lam = float(foo[i+1])
        if foo[i]=='-tau':    tau = float(foo[i+1])
        if foo[i]=='-gam':    gam = float(foo[i+1])
        if foo[i]=='-b':    b = int(float(foo[i+1]))
        if foo[i]=='-emph': emph = float(foo[i+1])
        if foo[i]=='-check': erchk=foo[i+1]
    
        
    savefile = savehead+'L'+str(lam)+'T'+str(tau)+'G'+str(gam)+'A'+str(emph)
    
    print('starting training with following parameters')
    print_params(r,lam,tau,gam,emph,ITERS)
    print('there are a total of {} words, and {} time points'.format(nw,T))
    
    print('X*X*X*X*X*X*X*X*X')
    print('initializing')
    
    #Ulist,Vlist = utils.initvars(nw,T,r, trainhead)
    Ulist,Vlist = utils.import_static_init(T, r)
    print(Ulist)
    print(Vlist)
    print('getting batch indices')
    if b < nw:
        b_ind = utils.getbatches(nw,b)
    else:
        b_ind = [range(nw)]
    
    import time
    start_time = time.time()
    # sequential updates
    for iteration in range(ITERS):  
        print_params(r,lam,tau,gam,emph,ITERS)
        try:
            Ulist = pickle.load(open( "%sngU_iter%d.p" % (savefile, iteration), "rb" ) )
            Vlist = pickle.load(open( "%sngV_iter%d.p" % (savefile, iteration), "rb" ) )
            print('iteration %d loaded succesfully' % iteration)
            continue
        except FileNotFoundError:
            pass
        loss = 0
        # shuffle times
        if iteration == 0: times = T
        else: times = np.random.permutation(T)
        
        for t in range(len(times)):   # select a time
            print('iteration %d, time %d' % (iteration, t))
            f = trainhead + str(T.start+t*aggregation_interval) + '.csv'
            print(f)
            
            """
            try:
                Ulist = pickle.load( open( "%sngU_iter%d_time%d_tmp.p" % (savefile,iteration,t), "rb" ) )
                Vlist = pickle.load( open( "%sngV_iter%d_time%d_tmp.p" % (savefile, iteration,t), "rb" ) )
                times = pickle.load( open( "%sngtimes_iter%d_time%d_tmp.p" % (savefile, iteration,t), "rb" ) )
                print('iteration %d time %d loaded succesfully' % (iteration, t))
                continue
            except(IOError):
                pass
            """
            
            pmi = utils.getmat(f,nw,False)
            for j in range(len(b_ind)): # select a mini batch
                print('%d out of %d' % (j,len(b_ind)))
                ind = b_ind[j]
                ## UPDATE V
                # get data
                pmi_seg = pmi[:,ind].todense()
                
                if t==0:
                    vp = np.zeros((len(ind),r))
                    up = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vp = Vlist[t-1][ind,:]
                    up = Ulist[t-1][ind,:]
                    iflag = False

                if t==len(T)-1:
                    vn = np.zeros((len(ind),r))
                    un = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vn = Vlist[t+1][ind,:]
                    un = Ulist[t+1][ind,:]
                    iflag = False
                Vlist[t][ind,:] = utils.update(Ulist[t],emph*pmi_seg,vp,vn,lam,tau,gam,ind,iflag)
                Ulist[t][ind,:] = utils.update(Vlist[t],emph*pmi_seg,up,un,lam,tau,gam,ind,iflag)
            
                
            #pickle.dump(Ulist, open( "%sngU_iter%d_time%d_tmp.p" % (savefile,iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
            #pickle.dump(Vlist, open( "%sngV_iter%d_time%d_tmp.p" % (savefile, iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
            #pickle.dump(times, open( "%sngtimes_iter%d_time%d_tmp.p" % (savefile, iteration,t), "wb" ) , pickle.HIGHEST_PROTOCOL)
       
                
            ####  INNER BATCH LOOP END
                
        # save
        print('time elapsed = ', time.time()-start_time)
       

        pickle.dump(Ulist, open( "%sngU_iter%d.p" % (savefile, iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
        pickle.dump(Vlist, open( "%sngV_iter%d.p" % (savefile, iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
