import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt


# I - les fonctions nécessaires et liées au modèle de Lotka-Volterra (modèle étudié)

def solutionTheoriqueLV(theta = [2,1,4,1], x0 = [5,3], pos = [221, 223], titre = ["",""]):
    """le système d'equadiff est de la forme :
    
        dS
        -- = S(alpha - beta * W)
        dt
    et
        dW
        -- = -W(gamma - delta * S)
        dt
    
    avec S le nombre de proies et W le nombre de prédateur
    
    METHODE D'EULER, Runge Kutta sûrement meilleure
    On résout sur [0,2]
    """
    alpha, beta, delta, gamma = theta
    h = 2/1000
    S=[x0[0]]
    W=[x0[1]]
    for i in range (1,1001):
        S.append(S[-1] + h * S[-1] * (alpha - beta * W[-1]))
        W.append(W[-1] - h * W[-1] * (delta - gamma * S[-2]))
    abscisse = np.linspace(0,2,1001)
    plt.figure(1)
    plt.subplot(pos[0])                        #graphe pour S
    plt.title(titre[0])
    plt.plot(abscisse, np.array(S), "b")
    plt.subplot(pos[1])
    plt.title(titre[1])                    #on met dans la deuxième ligne la courbe de W
    plt.plot(abscisse, np.array(W), "b")
    plt.show()
    return(S,W)

def observation(ecartTypeBruit = 0.5, nombre = 11, pos=[[221, 223]], solutionTheorique = solutionTheoriqueLV):
    """ on modélise les observations comme la donnée théorique sur laquelle on applique une certaine incertitude modélisé par une loi normale N(0, (ecartTypeBruit)**2)
    la variable pos sert à savoir où placer les observations
    """
    S,W = solutionTheorique([2, 1, 4, 1], [5,3], pos[0], ["S solution théorique","W solution théorique"])
    obsS = []
    obsW = []
    j = int(1000/(nombre-1))
    for i in range (0,nombre):
        obsS.append(S[i*j] + random.gauss(0, ecartTypeBruit)) 
        obsW.append(W[i*j] + random.gauss(0, ecartTypeBruit))
    abscisse = np.linspace(0,2,nombre)
    for i in pos:
        plt.subplot(i[0])
        plt.plot(abscisse, np.array(obsS), "r*")
        plt.subplot(i[1])
        plt.plot(abscisse, np.array(obsW), "r*")
    plt.show()
    return(obsS, obsW)

def fLV(S, W, theta):
    """la fonction f dans le modèle de Lotka-Volterra"""
    return(S*(theta[0] - theta[1]*W), -W*(theta[2] - theta[3]*S))
    
def SLV(Y, theta, nombre):
    """on calcule ici S pour le modèle de Lotka-Volterra"""
    deltat = 2/(nombre-1)
    return(sum([(Y[0][i] - Y[0][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[0]*deltat)**2 for i in range(1, nombre)]) + sum([(Y[1][i] - Y[1][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[1]*deltat)**2 for i in range(1, nombre)]))

def rLV(Y, theta, nombre):
    deltat = 2/(nombre-1)
    return(np.array([sqrt((Y[0][i] - Y[0][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[0]*deltat)**2 + (Y[1][i] - Y[1][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[1]*deltat)**2) for i in range(1, nombre)]))

def JacLV(Y, theta, nombre, r):
    deltat = 2/(nombre-1)
    return(np.array([[-1/r[i]*Y[0][i]*(Y[0][i+1]-Y[0][i]-Y[0][i]*(theta[0] - theta[1]*Y[1][i])*deltat),1/r[i]*Y[0][i]*Y[1][i]*(Y[0][i+1]-Y[0][i]-Y[0][i]*(theta[0] - theta[1]*Y[1][i])*deltat), 1/r[i]*Y[1][i]*(Y[1][i+1] - Y[1][i] + Y[1][i]*(theta[2] - theta[3]*Y[0][i])*deltat), -1/r[i]*Y[0][i]*Y[1][i]*(Y[1][i+1] - Y[1][i] + Y[1][i]*(theta[2] - theta[3]*Y[0][i])*deltat) ] for i in range(nombre-1)])*deltat)

def secondTermeHessienneLV(obs, nombre):
    matrice = np.zeros((4,4))
    for i in range(nombre-1):
        matrice = matrice + np.array([[obs[0][i]**2,-obs[0][i]**2*obs[1][i],0,0],[-obs[0][i]**2*obs[1][i], obs[0][i]**2*obs[1][i]**2, 0, 0],[0,0, obs[1][i]**2, -obs[0][i]*obs[1][i]**2],[0,0,-obs[0][i]*obs[1][i]**2, obs[0][i]**2*obs[1][i]**2]])
    deltat = 2*(nombre-1)
    return(matrice*(deltat)**2)

def gradientSLV(Y, theta, nombre = 11):
    """on calcule ici le gradient de S dans le modèle de Lotka-Volterra, pour cela on calcule chacune des dérivés partielles"""
    deltat = 2/(nombre-1)
    a = -2*sum([(Y[0][i] - Y[0][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[0]*deltat)*deltat*Y[0][i-1] for i in range(1,len(Y[0]))])
    b = +2*sum([(Y[0][i] - Y[0][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[0]*deltat)*deltat*Y[0][i-1]*Y[1][i-1] for i in range(1,len(Y[0]))])
    c = +2*sum([(Y[1][i] - Y[1][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[1]*deltat)*deltat*Y[1][i-1] for i in range(1,len(Y[0]))])
    d = -2*sum([(Y[1][i] - Y[1][i-1] - fLV(Y[0][i-1], Y[1][i-1], theta)[1]*deltat)*deltat*Y[1][i-1]*Y[0][i-1] for i in range(1,len(Y[0]))])
    return(a,b,c,d)


#II - Méthode de la descente en gradient pour la tester il suffit de lancer "testMethodeDescenteEnGradientLV()"

def methodeDescenteEnGradient(obs, theta0, nombre, pos, titre, gradientS = gradientSLV, S = SLV, solutionTheorique = solutionTheoriqueLV, pas = 10**(-3)):
    """cette fonction sert à appliquer la méthode de la descente en gradient"""
    theta = theta0                                                              #initialisation de theta
    arret = 42
    pas = 10**(-3)
    while arret > 0.001:                                                        #Notre condition d'arrêt est quand la norme du gradient de S est inférieur (ou égal) à 0.001 (donc qu'on arrive sur un plateau/partie plate)
        g = gradientS(obs, theta, nombre)                                       #Si le gradient de S à une dépendance en l'écart-type des observations il faut rajouter cette variable
        alpha = 0
        min = S(obs, [theta[i] - alpha*g[i] for i in range(len(g))], nombre)    #idem (vis à vis de l'écart-type)
        condition = min
        while condition <= min:                                                 #cette boucle sert à chercher le meilleur alpha (ie : l'argmax de S(theta - alpha*gradient(S)),
                                                                                #pour cela on va augmenter apha petit à petit (du pas) jusqu'a ce que S n'augmente plus (c'est la condition d'arrêt).
            min = condition
            alpha += pas
            condition = S(obs, [theta[i] - alpha*g[i] for i in range(len(g))], nombre)      #idem (vis à vis de l'écart-type)
        arret = sum([abs(i) for i in g])                                        #on calcule la norme 1 du gradient pour notre condition d'arrêt
        print(arret)                                                            #permet de suivre la progression du programme
        theta = [theta[i] - (alpha-pas)*g[i] for i in range(len(g))]            #on applique la formule de récurrence
    solutionTheorique(theta, [obs[i][0] for i in range(len(obs))], pos, titre)               #on trace les résultats
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode

def testMethodeDescenteEnGradientLV(ecartTypeBruit = 0.5, nombre = 101, theta0 = [1.5, 0.5, 3.5, 0.5], thetaTheorique = [2,1,4,1], X0 = [5,3], pos = [[221,223], [222, 224]], titre = [["S solution théorique","W solution théorique"],["S par la méthode de descente en gradient", "W par la méthode de descente en gradient"]], gradientS=gradientSLV, S = SLV, solutionTheorique = solutionTheoriqueLV, pas = 10**(-3)):
    """cette fonction a été pensée pour que notre code assez indépendant du systeme d'equation que l'on cherche à résoudre, et ici on résout le problème de Lotka-Volterra (LV), il y a beaucoup de paramètres mais objectivement peu servent vraiment :
    ecartTypeBruit et nombre servent à générer des observations
    theta0 à avoir une valeur pour initialiser notre algorithme
    thetaTheorique et X0 à pouvoir retracer la solution théorique (donc c'est de la déco)
    pos et titre c'est pour les graphes
    gradientS, S et solutionTheorique c'est pour savoir quel est le système d'équation que l'on utilise
    pas c'est le pas utilisé pour dans la recherche du meilleur alpha dans l'algo
    """
    obs = observation(ecartTypeBruit, nombre, pos)                              #on génère des observations
    solutionTheorique(thetaTheorique, X0, pos[0], titre[0])                               #on retrace la courbe théorique (elle est parfois cachée sous les observations)
    return(methodeDescenteEnGradient(obs, theta0, nombre, pos[-1], titre[1], gradientS, S, solutionTheorique, pas))    #on renvoie le théta trouvé à l'aide de cette méthode


#III - Méthode de Gauss-Newton et de Newton pour la tester il suffit de lancer "testMethodesDeNewtonLV()"

def methodeGaussNewton(obs, theta0, nombre, pos, titre, nombreDeRec, solutionTheorique = solutionTheoriqueLV, rFonction = rLV, Jacob = JacLV):
    """ ici on présente la méthode de Gauss-Newton"""
    theta = theta0                                                              #on initialise theta
    for i in range(nombreDeRec):
        r = rFonction(obs,theta, nombre)                                        #on calcule r
        Jac = Jacob(obs, theta, nombre, r)                                      #on calcule la jacobienne de r
        matrice = np.dot(np.linalg.inv(np.dot(Jac.T, Jac)),Jac.T)
        theta=theta - np.dot(matrice,r.T)
        theta0 = theta0 - np.dot(matrice,r.T)                                   #on applique la formule de récurrence
    solutionTheorique(theta, [obs[i][0] for i in range(len(obs))], pos, titre)               #on trace les résultats
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode

def methodeNewton(obs, theta0, nombre, pos, titre, nombreDeRec, solutionTheorique = solutionTheoriqueLV, rFonction = rLV, Jacob = JacLV, secondTermeHessienne = secondTermeHessienneLV):
    """ ici on présente la méthode de Newton sans l'approximation de Gauss-Newton"""
    theta = theta0                                                              #on initialise theta
    deltat = 2/(nombre-1)
    matrice = secondTermeHessienne(obs, nombre)                                 #!!!!  le second terme de la hessienne ne dépend que des observations dans notre cas, donc je l'ai mis ici,
                                                                                #!!!!  si ce n'est pas le cas il faut écrire cette ligne dans la boucle for qui suit
    for i in range(nombreDeRec):
        r = rFonction(obs, theta, nombre)                                       #on calcule r
        Jac = Jacob(obs, theta, nombre, r)                                      #on calcule la jacobienne de r
        mat = np.dot(np.linalg.inv(matrice + np.dot(Jac.T, Jac)),Jac.T)         
        theta = theta - np.dot(mat,r.T)                                         #on applique la fomrile de récurrence
    solutionTheorique(theta, [obs[i][0] for i in range(len(obs))], pos, titre)               #on trace la courbe
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode

def testMethodesDeNewtonLV(ecartTypeBruit = 0.5, nombre = 11, theta0 = [1.5, 0.5, 3.5, 0.5], thetaTheorique = [2,1,4,1], X0 = [5,3], pos = [[231,234], [232, 235], [233, 236]], titre = [["S solution théorique","W solution théorique"],["S par la méthode de Gauss-Newton", "W par la méthode de Gauss-Newton"],["S par la méthode de Newton", "W par la méthode de Newton"]], nombreDeRec = 1000, solutionTheorique = solutionTheoriqueLV, rFonction = rLV, Jacob = JacLV, secondTermeHessienne = secondTermeHessienneLV):
    """cette fonction a été pensée pour que notre code assez indépendant du systeme d'equation que l'on cherche à résoudre, et ici on résout le problème de Lotka-Volterra (LV), il y a beaucoup de paramètres mais objectivement peu servent vraiment :
    ecartTypeBruit et nombre servent à générer des observations
    theta0 à avoir une valeur pour initialiser notre algorithme
    thetaTheorique et X0 à pouvoir retracer la solution théorique (donc c'est de la déco)
    pos et titre c'est pour les graphes
    nombreDeRec correspond au nombre de fois que l'on va appliquer la formule de récurrence
    solutionTheorique, rFonction, Jacob et secondTermeHessienne est lié au système d'équation que l'on étudie
    """
    obs= observation(ecartTypeBruit, nombre, pos)
    solutionTheorique(thetaTheorique, X0, pos[0], titre[0])                     #on retrace la courbe théorique (elle est parfois cachée sous les observations)
    a = methodeGaussNewton(obs, theta0, nombre, pos[1], titre[1], nombreDeRec, solutionTheorique, rFonction, Jacob)
    b = methodeNewton(obs, theta0, nombre, pos[2], titre[2], nombreDeRec, solutionTheorique, rFonction, Jacob, secondTermeHessienne)
    return(a,b)                                                                 #on applique les deux méthodes, on trace les rendus et on renvoie le paramètre calculé pour chacune de ces méthodes.
    
    
# IV - Si on veut tester les 4 méthodes il suffit de lancer test()

def test(ecartTypeBruit = 0.5, nombre = 11, theta0 = [1.5, 0.5, 3.5, 0.5], thetaTheorique = [2,1,4,1], X0 = [5,3], pos = [[241,245], [242, 246], [243, 247], [244, 248]], titre = [["S solution théorique","W solution théorique"], ["S par la méthode de descente en gradient", "W par la méthode de descente en gradient"], ["S par la méthode de Gauss-Newton", "W par la méthode de Gauss-Newton"], ["S par la méthode de Newton", "W par la méthode de Newton"]], nombreDeRec = 1000, gradientS=gradientSLV, S = SLV, solutionTheorique = solutionTheoriqueLV, rFonction = rLV, Jacob = JacLV, secondTermeHessienne = secondTermeHessienneLV, pas = 10**(-3)):
    """c'est un mélange de tout"""
    obs= observation(ecartTypeBruit, nombre, pos)
    solutionTheorique(thetaTheorique, X0, pos[0], titre[0])                     
    a = methodeDescenteEnGradient(obs, theta0, nombre, pos[1], titre[1], gradientS, S, solutionTheorique, pas)
    b = methodeGaussNewton(obs, theta0, nombre, pos[2], titre[2], nombreDeRec, solutionTheorique, rFonction, Jacob)
    c = methodeNewton(obs, theta0, nombre, pos[3], titre[3], nombreDeRec, solutionTheorique, rFonction, Jacob, secondTermeHessienne)
    return(a,b,c)   
    
