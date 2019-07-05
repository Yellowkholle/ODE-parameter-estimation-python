import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal, gamma


def GPcov(s,t,ppar):
    """la fonction de covariance choisie pour notre prior phi (le processus gaussien)
    """
    return(ppar[0]*np.exp(-ppar[1]*(s-t)**2))

def LV(t,x, theta):
    """fonction f dans la formule de l'EDO
    """
    return(float(x[0]*(theta[0] - theta[1]*x[1])), float(-theta[1]*(theta[2] - theta[3]*x[0])))

def dGPcov(s,t,ppar):
    """la dérivé de la covariance choisie
    """
    return(-2*ppar[1]*(s-t)*GPcov(s,t,ppar))

def ddGPcov(s,t,ppar):
    """la dérivée seconde de la covariance choisie
    """
    return((2*ppar[1] - 4*ppar[1]**2*(s-t)**2)*GPcov(s,t,ppar))

def ode_ComGP_solve_X0(Data, ODE, GP, Iteration, X0):
    """
        On applique un échantillonnage de Gibbs, comme j'ai globalement traduit le code
        de David Barber que l'on peut trouver sur github.com/odegp/code et que ses commentaires sont assez clairs je les ai souvent repris (ce sont ceux en anglais :) )
    
        %% ODE parameter estimation with common GP settings (Sampling X0 is incorporated)
        
        %%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Data:
        %      Data.y: observations
        %      Data.samptime: the sample time
        
        % ODE:
        %      ODE.fun: ODE functions
        %      ODE.num: the number of the parameters
        %      ODE.discrete: the discretized ranges of the parameters
        %      ODE.initial: initial values of ODE parameters
        %      ODE.prior: prior for ODE parameters 
        
        % GP:
        %      GP.fun: GP covariance function: c(t,t')
        %      GP.fun_d: GP derivative d{c(t,t')}/dt
        %      GP.fun_dd: GP derivative d^2{c(t,t')}/dtdt'
        %      GP.num: the number of the GP hyperparameters (we put the noise std to the end of the hyperparameter vector)
        %      GP.discrete: the discretized ranges of the GP hyperparameters
        %      GP.inital_X: initial mean, std of X, and the number of discretized bins for X
        %      GP.inital: initial values of GP hyperparameters
        %      GP.prior: prior for GP hyperparameters
        
        % Iter:
        %      Iteration.total: the number of the total iterations
        %      Iteration.sub:   the number of the sub iterations
        
        % X0:
        %      X0.indicator: indicator to show whether X0 is specified, here X0.indicator=1
        %      X0.discrete: the discretized ranges of X0  
        %      X0.initial: intial values of x0;
        %      X0.prior: prior of x0
        
        
        %%%%%%%%%%%%%%%%%%% Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % samptheta: samples of ODE parameters
        % samphyper: samples of GP hyperparameters
        % SAMPLEX: samples of X
    """
    
    #initialisation
    ODEfun, ODEnum, ODEdiscrete, ODEinitial, ODEprior = ODE
    GPfun, GPfun_d, GPfun_dd, GPnum, GPdiscrete, GPprior, GPinitial, GPinitial_X = GP
    Iterationtotal, Iterationsub = Iteration
    X0discrete, X0initial, X0prior = X0
    
    
    #Load Data and Sampletime
    y, samptime = Data
    TTT = len(samptime)
    
    #Discratized Grid
    #ODE
    ODE_par = []
    for i in range(ODEnum):
        ODE_par.append(ODEdiscrete[i])
        
    #GP
    GP_par = []
    for i in range(GPnum):
        GP_par.append(GPdiscrete[i])
        
    #X0
    X_0 = []
    for i in range(len(y[0])):
        X_0.append(X0discrete[i])
    
    # Initialization
    #ODE
    odepar = ODEinitial
    
    #GP
    MeanX, StdX = GPinitial_X[:2]
    SX = MeanX
    
    gppar = GPinitial
    
    #X0
    x0 = X0initial
    
    samptheta = []
    samphyper = []
    samplex0 = []
    SAMPLEX = []
    
    # Gibbs sampling
    for iter in range(Iterationtotal):
        print(iter, odepar)                                                     #permet de suivre l'avancement
        #------------------------- Sampling Parameters ------------------------
        meany = np.tile(x0, (TTT,1))
        
        for subiter in range(Iterationsub):
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #+++ Sample GP hyperparameters, fix ODE parameter, noise std, X and X0
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            #+++++ Note before sapling hyperparameters +++++++++++++
            # Since X and ODE parameters are fixed, dotX are fixed
            
            # on va dans un premier temps fixer les paramètres de l'EDO (theta), sigma, X et X0
            # remarque : theta et X sont fixés, donc Xpoint = f(X, theta) l'est aussi
            # donc p(theta), pODE(Xpoint | X, theta) sont fixés
            # il nous reste donc p(phi), pGP(Y |Xpoint, phi) et pGP(X|phi) à calculer pour obtenir la loi jointe (les autres termes sont constants)
            dotx = []                                                           #on va calculer Xpoint (graĉe à l'ODE)
            for i in range(TTT):
                dotx.append(ODEfun(samptime[i], SX[i], odepar))
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            #+++ Sample GP hyperparameters (on va donc d'abord s'intéresser à sigmax et l
            #pour cela on va calculer toutes les lois jointes (en fonction de ces paramètres, qui vont donc varier)
            #que l'on va stocker dans pgp
            for k in range(GPnum-1):
                IT = GP_par[k]
                pgp = []
                
                for j in range(len(IT)):
                    gppar[k] = IT[j]
                    
                    Cxx = np.zeros((TTT, TTT))                                  #on calcule les matrices de covariance et ses "dérivées" nécssaire pour calculer pGP(Y | Xpoint)
                    dC = np.zeros((TTT, TTT))
                    ddC = np.zeros((TTT, TTT))
                    for s in range(TTT):
                        for t in range(TTT):
                            Cxx[s,t] = GPfun(s,t,gppar)
                            dC[s,t] = GPfun_d(s,t,gppar)
                            ddC[s,t] = GPfun_dd(s,t,gppar)
                    Cxx = 1/2*(Cxx+Cxx.T)
                    
                    #p(y | dotx, x0) ce sera p1
                    Mygxdot = np.dot(dC.T, np.linalg.inv(ddC))
                    
                    meandotx = np.dot(Mygxdot, dotx) + meany
                    
                    Cyy =Cxx + gppar[-1]**2*np.eye(TTT)
                    Cygxdot = Cyy - np.dot(Mygxdot, dC)
                    Cygxdot = 1/2* (Cygxdot + Cygxdot.T)
                    
                    p1 = 1
                    p2 = 1                                                      # p2 = p(x|phi, x0)
                    for i in range(len(y[0])):
                        p1 = p1*multivariate_normal.pdf(y[:,i].T, meandotx[:,i].T, Cygxdot)
                        p2 = p2*multivariate_normal.pdf(SX[:,i].T, meany[:,i].T, Cxx)
                    p3 = GPprior[k](gppar[k])                                   # p3 = p(phi)
                    pgp.append(p1*p2*p3)
                
                totPoids = sum(pgp)
                gppar[k] = float(np.random.choice(IT, 1, p=[i/totPoids for i in pgp]))      #on va donc tirer au sort de nouveaux paramètres sigmax et l
                IT, pgp = [], []
             
             #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
             #+++ Sample noise std, fix GP hyperparameters, ODE parameter, X and X0
             #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
             
             #+++ Note before sample noise std ++++++++++
             
             # Since X and ODE parameters are fixed, dotX are still fixed as before
             # Since X0 is fixed, meany is fixed as before
             # Since GP hyperparameters are fixed now, the covariance matrice are fixed
             
             # on va dans cette partie essayer de générer le dernier paramètre de phi : sigma en fixant tous les autres paramètres
             # donc il ne nous reste que p(phi), pGP(Y | Xpoint, phi) à calculer. pGP(X|phi) ne dépend que de sigmax et de l.
            
            #on calcule les matrices de covariance et ses "dérivées" nécessaires pour calculer pgP(Y |Xpoint)
            Cxx = np.zeros((TTT, TTT))                                          
            dC = np.zeros((TTT, TTT))
            ddC = np.zeros((TTT, TTT))
            for s in range(TTT):
                for t in range(TTT):
                    Cxx[s,t] = GPfun(s,t,gppar)
                    dC[s,t] = GPfun_d(s,t,gppar)
                    ddC[s,t] = GPfun_dd(s,t,gppar)
            
            #Hence Mean of p(y |dotx, x0) is now fixed
            Mygxdot = np.dot(dC.T, np.linalg.inv(ddC))
            meandotx = np.dot(Mygxdot, dotx)+ meany
            
            #Hence Covariance of p(y |dotx, x0) is now fixed except sigma**2 Id
            Cmiddle = Cxx - np.dot(Mygxdot, dC)
            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            # Sampling Sigma
            IT = GP_par[-1]
            psg = []
            for j in range(len(IT)):            #on va ici parcourir toutes les valeurs possibles de sigma
                gppar[-1] = IT[j]
                
                #p(y|dotx, x0, phi)
                Cygxdot = Cmiddle + gppar[-1]**2*np.eye(TTT)
                Cygxdot = 1/2*(Cygxdot + Cygxdot.T)
                
                p1 = 1
                for i in range(len(y[0])):
                    p1 = p1*multivariate_normal.pdf(y[:,i].T, meandotx[:,i].T, Cygxdot)
                
                #p(phi)
                p2 = GPprior[-1](gppar[-1])
                psg.append(p1*p2)
            
            totPoids = sum(psg)
            gppar[GPnum-1] = float(np.random.choice(IT, 1, p=[i/totPoids for i in psg]))    #on va tirer au sort un sigma
            IT, psg = [], []
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #+++ Sample ODE parameter, fix GP hyperparameters, noise std, X and X0
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            #++ Note before sampling ODE parameter +++++++++
            
            # Since GP hyperparameters are fixed, covariance matrice are still fixed
            # Since X0 is fixed, meany is fixed as before      
            # Hence Mygxdot=dC'*inv(ddC) in the Mean of p(y|dotx,x0) is still fixed
            # Since noise std is now fixed, Hence Covaraince of p(y|dotx,x0) is fixed
            
            # on va ici générer de nouvelles valeurs pour les paramètres de l'EDO (et donc theta), en fixant tous les autres (ie : phi, sigma, X et X0)
            # remarque : Xpoint n'est pas fixé car il dépend de theta
            # il ne nous reste que p(theta), pGP(Y|Xpoint, phi), pODE(Xpoint | X, theta) à calculer pour obtenir la loi jointe
            # on va cacher le calcul de pODE(Xpoint | X, theta), car il vaut Dirac(Xpoint-f(X, theta)), donc si le Xpoint est le bon cette proba vaut 1
            
            Cygxdot = Cmiddle + gppar[-1]**2*np.eye(TTT)               #!!! Cmiddle est il bien def ??
            Cygxdot = 1/2*(Cygxdot + Cygxdot.T)
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            #+++ Sampling ODE parameters
            for k in range(ODEnum):             #on va les faire les uns après les autres
                IT = ODE_par[k]
                pode = []
                for j in range(len(IT)):
                    odepar[k] = IT[j]
                    
                    dotx = []                   #il nous faut calculer Xpoint pour les valeurs de theta que l'on étudie
                    #p(y|dotx, x0)
                    for i in range(TTT):
                        dotx.append(ODEfun(samptime[i], SX[i], odepar))
                    dotx = np.array(dotx)
                    meandotx = np.dot(Mygxdot, dotx) + meany
                    
                    p1 = 1
                    for i in range(len(y[0])):
                        p1 = p1* multivariate_normal.pdf(y[:,i].T, meandotx[:,i].T, Cygxdot)
                    
                    #p(theta)
                    p2 = ODEprior[k](odepar[k])
                    
                    pode.append(p1*p2)
                    
                totPoids = sum(pode)
                odepar[k] = float(np.random.choice(IT, 1, p=[i/totPoids for i in pode]))        #on va tirer au sort de nouveaux paramètres théta
                IT, pode = [], []
        
        samptheta.append(odepar.copy())
        samphyper.append(gppar.copy())
        
        dotx = []
        
        #++++++++++++ SAmpling X0 along with dimension +++++++++++++
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++      CETTE ETAPE DEVRAIT ETRE FACULTATIVE EN AJOUTANT UN IF       ++
        #++ MAIS SINON NE PROPOSER QU'UNE VALEUR DANS LA DISCRETiSATION DE X0 ++
        #++                   DEVRAIT THEORIQUEMENT SUFFIRE                   ++
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for i in range(TTT):
            dotx.append(ODEfun(samptime[i], SX[i], odepar))
        
        # Mygxdot=dC'*inv(ddC) in the Mean of p(y|dotx,x0) is still fixed 
        Mean_middle = np.dot(Mygxdot, dotx)
        
        # Covaraince of p(y|dotx,x0) is still fixed
        
        for subiter in range(Iterationsub):
            
            for DIM in range(len(y[0])):
                
                IT1 = X_0[DIM]
                px0 = []
                
                for j in range(len(IT1)):
                    
                    x0[DIM] = IT1[j]
                    
                    #p1 = p(y|dotx, x0)
                    #p2 = p(x|phi, x0)
                    
                    meany = np.tile(x0, (TTT, 1))
                    meandotx = Mean_middle + meany
                    
                    p1 = 1
                    p2 = 1
                    for i in range(len(y[0])):
                        p1 = p1*multivariate_normal.pdf(y[:,i].T, meandotx[:,i].T, Cygxdot)
                        p2 = p2*multivariate_normal.pdf(SX[:,i].T, meany[:,i].T, Cxx)
                    
                    #p3 = p(X0)
                    p3 = X0prior[DIM](x0[DIM])
                    
                    px0.append(p1*p2*p3)
                    
                totPoids = sum(px0)
                x0[DIM] = float(np.random.choice(IT1, 1, p=[i/totPoids for i in px0]))
                IT1, px0 = [], []
                
        samplex0.append(x0.copy())
        
        #+++++++++ Sampling X along with dimension ++++++++++++
        
        #+++++++++ Note before sampling  X ++++++++++++++++++++      
        
        # Since GP hyperparameters and noise std are fixed,         
        # Hence Mygxdot=dC'*inv(ddC) in the Mean of p(y|dotx) is fixed
        # Hence Covaraince of p(y|dotx) is fixed
        
        # on va ensuite dans cette partie modifier X, en fixant tous les autres paramètres (Xpoint n'est pas fixé)
        # il nous faut donc calculer p(Y|Xpoint, phi), pODE(Xpoint |X, theta) et p(X|phi)
        # meme remarque que quand on a chercher theta pour ce qui est de pODE(Xpoint |X, theta)
        
        # Since xo is fixed, then meany is fixed now
        
        for subiter in range(Iterationsub):
            for Tstep in range(TTT):
                for DIM in range(len(y[0])):
                    IT1 = np.linspace(MeanX[Tstep, DIM] - 3*StdX[Tstep, DIM], MeanX[Tstep, DIM] + 3*StdX[Tstep, DIM], GPinitial_X[2])
                    px1 = []
                    
                    for j in range(len(IT1)):
                        SX[Tstep, DIM] = IT1[j]
                        
                        # on va donc calculer Xpoint
                        dotx = []
                        for i in range(TTT):
                            dotx.append(ODEfun(samptime[i], SX[i], odepar))
                            
                        meandotx = np.dot(Mygxdot, dotx) + meany
                        
                        # p1S = p(y|dotx, x0)
                        # p2S = p(x|phi, x0)
                        p1S = 1
                        for i in range(len(y[0])):
                            p1S = p1S * multivariate_normal.pdf(y[:,i].T, meandotx[:,i].T, Cygxdot)
                        p2S = multivariate_normal.pdf(SX[:,DIM].T, meany[:,DIM].T, Cxx)
                        
                        px1.append(p1S*p2S)
                    totPoids = sum(px1)
                    SX[Tstep, DIM] = float(np.random.choice(IT1, 1, p=[i/totPoids for i in px1]))
                    IT1, px1 = [], []
        SAMPLEX.append(SX.copy())
    return(samptheta, samphyper, samplex0, SAMPLEX)             #on va renvoyer tous les paramètres qui ont été générés dans les 5 étapes, qui pourront ensuite être traités si besoin

def FitGP(Data, GP):
    """Le but de cette focntion est d'exprimer un X0 pour pouvoir l'initialiser,
    Pour cela on cherche X ~ p(X| Y, phi) où Y est la vecteur des observations et phi le PG.
    """
    
    GPfun, GPfun_d, GPfun_dd, GPnum, GPdiscrete, GPprior, GPinitial = GP
    
    y, samptime = Data                                                          #on récupère les observations
    
    TTT = len(samptime)
    meany = np.tile(np.mean(y,0), (TTT, 1))
    
    MI = []
    
    for i in range(GPnum):
        MI.append(GPdiscrete[i])
    
    nID = 3                                                                     #nombre de paramètres de phi (le GP) à déterminer
    
    ID = np.array([[i,j,k] for i in range(len(GPdiscrete[0])) for j in range(len(GPdiscrete[1])) for k in range(len(GPdiscrete[2]))])               #ensemble des triplets de paramètres de phi possibles
    
    
    I_NUM = len(ID)                                                             #nombre de combianaison de paramètres possibles
    
    W_HYP = []                                                                  #on va stocker ici pour chaque combianaison de paramètre la probabilité d'obtenir les observations sachant les observations que l'on a faite (p(x, phi|y))
    
    for i in range(I_NUM):
        WP = []                                                                 #permettra de calculer p(phi) en multipliant p(sigmax), p(l) et p(sigma) grâce aux priors
        ppar = []                                                               #valeurs des paramètres que l'on étudie
        
        for j in range(GPnum):
            WP.append(GPprior[j](MI[j][ID[i,j]]))
        
        for j in range(GPnum-1):
            ppar.append(MI[j][ID[i,j]])
        
        Cxx = np.zeros((TTT, TTT))                                              #matrice de covariance
        for s in range(TTT):
            for t in range(TTT):
                Cxx[s,t] = GPfun(s,t,ppar)
        Cxx = 1/2*(Cxx+Cxx.T)
        
        W_y = 1                                                                 #sert à calculer p(x|y, phi)
        
        for dimension in range(len(y[0])):
            W_y = W_y * multivariate_normal.pdf(y[:,dimension].T, meany[:,dimension].T, Cxx + MI[GPnum-1][ID[i,GPnum-1]]**2*np.eye(TTT))
        
        W_HYP.append(W_y * np.prod(WP))                                         #le p(x, phi|y) dont on avai parlé
        
    totPoids = sum(W_HYP)
    ind_hyp = np.random.choice(I_NUM, 200, p=[i/totPoids for i in W_HYP])       #on tire donc au hasard 200 combianaisons de paramètres (ça permet d'avoir comme la densité de proba des combianaisons de paramètres possibles)
    
    SAM_X = np.zeros((200, 20, 11, 2))                                          #va servir à générer 4000 simulations possibles (20 pour chaque combinaisons de paramètres tiré au sort), cela permettra d'avoir une "vraie" densité possible de résultats possibles à partir desquels on peut choisir moyenne et ecart-type
    
    for i in range(200):                                                        #on va donc s'intéresser aux 200 combinaisons de paramètres possibles
        ppar = []
        
        for j in range(GPnum - 1):
            ppar.append(MI[j][ID[ind_hyp[i], j]])
        
        Cxx = np.zeros((TTT, TTT))                                              #matrice de covariance
        for s in range(TTT):
            for t in range(TTT):
                Cxx[s,t] = GPfun(s,t,ppar)
        Cxx = 1/2*(Cxx+Cxx.T)
        
        ppar.append(MI[GPnum-1][ID[ind_hyp[i], GPnum-1]])
        
        Cxy = Cxx                                                               #on va calculer les matrices nécessaires
        Cyy = Cxx + ppar[GPnum-1]**2*np.eye(TTT)
        Mxgy = np.dot(Cxy, np.linalg.inv(Cyy))
        MUX = meany + np.dot(Mxgy, (y - meany))
        #covariance
        Cxgy = Cxx - np.dot(Mxgy, Cxy.T)
        Cxgy = 0.5*(Cxgy + Cxgy.T)
        
        for X_NUM in range(20):                                                 #on génère 20 résultats possibles (pas avec l'equadiff, mais d'un point de vue probabiliste)
            
            sampx = np.zeros((TTT, len(y[0])))
            
            for dimension in range(len(y[0])):
                sampx[:, dimension] = (np.random.multivariate_normal(MUX[:,dimension].T, Cxgy).T)
            
            SAM_X[i, X_NUM] = sampx
    
    X = np.zeros((len(y[0]), TTT, 1))                                           #on réécrit la matrice différemment pour pouvoir calculer plus facilement moyenne et écart-type
    
    for i in range(200):
        for X_NUM in range(20):
            UX = np.zeros((len(y[0]), TTT, 1))
            for k in range(len(y[0])):
                for l in range(TTT):
                    UX[k,l] = np.array(SAM_X[i, X_NUM, l, k])
            X = np.concatenate((X, UX), 2)
    
    MeanX = []                                                                  #les noms semblent explicites : servent à avoir moyenne et écart-type des "x" générés
    StdX = []
    
    for i in range(len(y[0])):
        MeanX.append(np.mean(X[i][:,1:], 1))
        StdX.append(np.std(X[i][:,1:], 1))
    MeanX = np.array(MeanX)
    StdX = np.array(StdX)
    print(MeanX.T, StdX.T)                                                      #on les affiches, car c'est pas inintéressant après tout
    return(MeanX.T, StdX.T)

def demo_LV():
    """Cette application est l'application à lancer (et modifier) si 'lon veut tester notre algorithme sur le modèle de Lotka-Volterra, on y applique donc toutes les étapes :
    initialisation des priors, ...
    initialisation de X grâce à FitGP
    récolte des infomations obtenues grâce à l'échatnillonnage de Gibbs
    calcul de la moyenne des résultats obtenus que l'on va renvoyer
    """
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++   Initialisation    +++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    nombre = 11
    ecartTypeBruit = 0.5
    Data = [observation(ecartTypeBruit, nombre),np.linspace(0, 2, nombre)]
    ODE = [LV, 4, [np.linspace(1.5, 2.5, 11), np.linspace(0.5, 1.5, 11), np.linspace(3.5, 4.5, 11), np.linspace(0.5, 1.5, 11)], [1.5, 0.5, 3.5, 0.5], [gamma405, gamma405, gamma405, gamma405]]
    X0 = [[np.linspace(0, 10, 20), np.linspace(0, 10, 20)], [5, 3], [unif10, unif10]]            # à modifier si on a un X0
    GP = [GPcov, dGPcov, ddGPcov, 3, [np.linspace(0.1, 1, 10), np.linspace(5, 50, 10), np.linspace(0.1, 1, 10)], [unif100, unif100, gamma11], [1, 10, 0.5]]
    
    #on initialise X avec FitGP
    MeanX, StdX = FitGP(Data, GP)
    
    GPinitial_X = [MeanX, StdX, 20]
    
    #nombre de fois que l'on va utiliser l'échantillonnage de Gibbs
    Iteration = [100, 10]
    Iterationtotal, Iterationsub = Iteration
    
    GP.append(GPinitial_X)
    
    #on applique l'échantillonnage de Gibbs
    samptheta, samphyper, samplex0, SAMPLEX = ode_ComGP_solve_X0(Data, ODE, GP, Iteration, X0)
    
    #résultat recherché
    MUtheta = np.mean(samptheta, 0)
    STDtheta = np.std(samptheta, 0)
    
    #on va tracer la solution théorique que l'on trouve
    sol = solutionTheoriqueLV(MUtheta, [5,3], False)
    abscisse = np.linspace(0,2,1001)
    plt.subplot(211)                                                            #graphe pour S
    plt.plot(abscisse, sol[0], "g")
    plt.subplot(212)                                                         #on met dans la deuxième ligne la courbe de W                  
    plt.plot(abscisse, sol[1], "g")
    plt.show()
    
    return(MUtheta, STDtheta)
    
    
    
def solutionTheoriqueLV(theta = [2,1,4,1], x0 = [5,3],bool = True, pos = [221, 223], titre = ["",""]):
    """le système d'equadiff est de la forme :
    
        dS
        -- = S(alpha - beta * W)
        dt
    et
        dW
        -- = -W(gamma - delta * S)
        dt
    
    avec S le nombre de proies et W le nombre de prédateur
    theta = [alpha, beta, gamma, delta]
    X0 correspond à l'initialisation
    
    bool sert à savoir si ontrace les graphes (True) ou non (False)
    pos sert à savoir où l'on place les graphes sur la figure (c'est surtout un reste de Lotka Volterra version classiques)
    titre correspond aux titres que l'on affiche avec les graphes
    
    METHODE D'EULER, (Runge Kutta est sûrement meilleure)
    
    On résout sur [0,2]
    """
    alpha, beta, delta, gamma = theta
    h = 2/1000                                                                  # pas pour appliquer la méthode d'Euler
    S=[x0[0]]
    W=[x0[1]]
    for i in range (1,1001):                                                    # on applique la méthode d'Euler
        S.append(S[-1] + h * S[-1] * (alpha - beta * W[-1]))
        W.append(W[-1] - h * W[-1] * (delta - gamma * S[-1]))
    abscisse = np.linspace(0,2,1001)
    if bool:
        plt.figure(1)
        plt.subplot(pos[0])                                                     #graphe pour S
        plt.title(titre[0])
        plt.plot(abscisse, np.array(S), "b")
        plt.subplot(pos[1])                                                     #on met dans la deuxième ligne la courbe de W
        plt.title(titre[1])                    
        plt.plot(abscisse, np.array(W), "b")
        plt.show()
    return(S,W)

def observation(ecartTypeBruit = 0.5, nombre = 11, pos=[[211, 212]], solutionTheorique = solutionTheoriqueLV):
    """ on modélise les observations comme la donnée théorique sur laquelle on applique une certaine incertitude modélisé par une loi normale N(0, (ecartTypeBruit)**2)
    la variable pos sert à savoir où placer les observations
    """
    S,W = solutionTheorique([2, 1, 4, 1], [5,3], True, pos[0], ["S solution théorique","W solution théorique"])
    obsS = []
    obsW = []
    j = int(1000/(nombre-1))                                                    #on modifie le "pas" en fonction du nombre d 'observations que l'on veut générer
    for i in range (0,nombre):
        obsS.append(S[i*j] + random.gauss(0, ecartTypeBruit))                   #on ajoute du bruit aux solutions théoriques
        obsW.append(W[i*j] + random.gauss(0, ecartTypeBruit))
    abscisse = np.linspace(0,2,nombre)
    for i in pos:                                                               #on trace les graphes
        plt.subplot(i[0])
        plt.plot(abscisse, np.array(obsS), "r*")
        plt.subplot(i[1])
        plt.plot(abscisse, np.array(obsW), "r*")
    plt.show()
    return(np.array([obsS, obsW]).T)

def gamma405(x):
    """loi gamma de paramèetres 4 et 0.5, elle sert de prior à alpha, béta, gamma et delta dans notre exemple
    """
    return(gamma.pdf(x, a=4, scale =0.5))

def gamma11(x):
    """loi a priori (exponentielle) pour le paramètre sigma (le bruit de nos observations)"""
    return(np.exp(-x))

def unif10(x):
    """loi a priori (uniforme) pour les paramètres de phi (notre prior processus gaussien) sigmax et l, et de X0"""
    return(1/10)

def unif100(x):
    return(1/100)
