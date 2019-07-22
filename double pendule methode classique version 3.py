import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt
from scipy.optimize import differential_evolution


# I - les fonctions nécessaires et liées au modèle de Lotka-Volterra (modèle étudié)

def solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, pos = [221,223], titre = ["theta 1", "theta 2"]):
    h = 30/50000
    listeTheta1 = [theta1]
    listeTheta2 = [theta2]
    listeThetaPoint1 = [thetaPoint1]
    listeThetaPoint2 = [thetaPoint2]
    for i in range(1, 50001):
        a, b = fDoublePendule(m, l, g, listeTheta1[-1], listeTheta2[-1], listeThetaPoint1[-1], listeThetaPoint2[-1])
        listeThetaPoint1.append(listeThetaPoint1[-1] + h * a)
        listeThetaPoint2.append(listeThetaPoint2[-1] + h * b)
        listeTheta1.append(listeTheta1[-1] + h * listeThetaPoint1[-1])
        listeTheta2.append(listeTheta2[-1] + h * listeThetaPoint2[-1])
    abscisse = np.linspace(0,30,50001)
    plt.figure(1)
    plt.subplot(pos[0])                        #graphe pour S
    plt.title(titre[0])
    plt.plot(abscisse, np.array(listeTheta1), "b")
    plt.subplot(pos[1])
    plt.title(titre[1])                    #on met dans la deuxième ligne la courbe de W
    plt.plot(abscisse, np.array(listeTheta2), "b")
    plt.xlabel("m = " + str(m) + ", l = " + str(l) + ", g = " + str(g))
    plt.show()
    return(listeTheta1, listeTheta2)

def observationDoublePendule(m=8, l=1, g=9.81, theta1 = (np.pi)/4, theta2 = (np.pi)/4, thetaPoint1 = 0, thetaPoint2 = 0, ecartTypeBruit = 0.25, nombre = 101, pos = [[221, 223]]):
    """on modélise des observations pour les angles du double pendule
    """
    lTheta1, lTheta2 = solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    obsTheta1 = []
    obsTheta2 = []
    j = int(50000/(nombre-1))
    for i in range(0, nombre):
        obsTheta1.append(lTheta1[i*j] + random.gauss(0, ecartTypeBruit))
        obsTheta2.append(lTheta2[i*j] + random.gauss(0, ecartTypeBruit))
    abscisse = np.linspace(0, 30, nombre)
    for i in pos:
        plt.subplot(i[0])
        plt.plot(abscisse, np.array(obsTheta1), "r*")
        plt.subplot(i[1])
        plt.plot(abscisse, np.array(obsTheta2), "r*")
    plt.show()
    return(obsTheta1, obsTheta2)

def fDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    a = 1/((1 + m) - m * np.cos(theta1 - theta2)**2) * (- m * l * thetaPoint2**2 * np.sin(theta1 - theta2) - (1 + m)*g*np.sin(theta1) + m * np.cos(theta1 - theta2) * (g * np.sin(theta2) - thetaPoint1**2 * np.sin(theta1 - theta2)))
    b = 1/l * (thetaPoint1**2 * np.sin(theta1 - theta2) - g * np.sin(theta2) - a * np.cos(theta1 - theta2))
    return(a,b)

def SDoublePendule(param, obs, nombre):
    theta1, theta2 = obs
    m, l, g = param
    sol = 0
    deltat = 30 / (nombre-1)
    thetaPoint1, thetaPoint2 = [0], [0]
    for i in range(len(theta1)-1):
        thetaPoint1.append((theta1[i+1] - theta1[i])/deltat)
        thetaPoint2.append((theta2[i+1] - theta2[i])/deltat)
    for i in range(nombre-1):
        f1, f2 = fDoublePendule(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        sol += (thetaPoint1[i+1] - thetaPoint1[i] - deltat * f1)**2 + (thetaPoint2[i+1] - thetaPoint2[i] - deltat * f2)**2
    return(sol)

def gradientSDoublePendule(obs, param, nombre):
    """ici on cherche m1 et m2
    """
    m, l, g = param
    theta1, theta2 = obs
    deltat = 30 / (nombre-1)
    sol1 = 0
    sol2 = 0
    sol3 = 0
    thetaPoint1, thetaPoint2 = [0], [0]
    for i in range(len(theta1)-1):
        thetaPoint1.append((theta1[i+1] - theta1[i])/deltat)
        thetaPoint2.append((theta2[i+1] - theta2[i])/deltat)
    for i in range(len(theta1)-1):
        f1, f2 = fDoublePendule(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        sol1 += - 2 * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint1[i+1] - thetaPoint1[i] - deltat * f1) * deltat
        - 2 * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint2[i+1] - thetaPoint2[i] - deltat * f2) * deltat
        sol2 += - 2 * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint1[i+1] - thetaPoint1[i] - deltat * f1) * deltat
        - 2 * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint2[i+1] - thetaPoint2[i] - deltat * f2) * deltat
        sol3 += - 2 * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint1[i+1] - thetaPoint1[i] - deltat * f1) * deltat
        - 2 * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * (thetaPoint2[i+1] - thetaPoint2[i] - deltat * f2) * deltat
    return(sol1, sol2, sol3)            #mettre des 0 là où on connaît les valeurs

  
  
    
def df1DoublePenduledm(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    a = -m * l * thetaPoint2**2 * np.sin(theta1 - theta2) - (1 + m)*g*np.sin(theta1) + m * np.cos(theta1 - theta2) * (g * np.sin(theta2) - thetaPoint1**2 * np.sin(theta1 - theta2))
    b = - (1 - np.cos(theta1 - theta2)**2)/((1 + m) - m * np.cos(theta1 - theta2)**2)**2 * a + 1/((1 + m) - m * np.cos(theta1 - theta2)**2) * (-g * np.sin(theta1) - l*thetaPoint2**2*np.sin(theta1 - theta2) + np.cos(theta1 - theta2) * (g * np.sin(theta2) - thetaPoint1**2*np.sin(theta1 - theta2)))
    return(b)

def df1DoublePenduledl(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    b = - m * thetaPoint2**2 * np.sin(theta1 - theta2)/((1+m) - m*np.cos(theta1 - theta2)**2)
    return(b)
    
def df1DoublePenduledg(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    b = (m * np.cos(theta1 - theta2) * np.sin(theta2) - (1+m)*np.sin(theta1))/((1+m) - m*np.cos(theta1 - theta2)**2)
    return(b)

def df2DoublePenduledm(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    a = df1DoublePenduledm(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    b = - 1/l * np.cos(theta1 - theta2) * a
    return(b)

def df2DoublePenduledl(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    a = df1DoublePenduledl(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    f1, f2 = fDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    b = 1/l**2 * (-thetaPoint1**2 * np.sin(theta1 - theta2) + np.cos(theta1 - theta2) * f1 - l*np.cos(theta1 - theta2) * a + g*np.sin(theta2))
    return(b)

def df2DoublePenduledg(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2):
    a = df1DoublePenduledg(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    b = -1/l*np.cos(theta1 - theta2)*a - np.sin(theta2)/l
    return(b)



def d2f1dm2(theta1, theta2, thetaPoint1, thetaPoint2, params):
    m, l, g = params
    sol = -2 * (-l*thetaPoint2**2*np.sin(theta1 - theta2) - g*np.sin(theta1) + np.cos(theta1 - theta2)*(g*np.sin(theta2) - thetaPoint1**2*np.sin(theta1 - theta2)))*(1 - np.cos(theta1 - theta2)**2)/((1 + m) - m * np.cos(theta1 - theta2)**2)**2 - 2 * (1 - np.cos(theta1 - theta2)**2)**2*(m*l*thetaPoint2**2*np.sin(theta1 - theta2) + (1+m)*g*np.sin(theta1) - m*np.cos(theta1 - theta2)*(g*np.sin(theta2) - thetaPoint1**2*np.sin(theta1 -theta2)))/((1 + m) - m * np.cos(theta1 - theta2)**2)**3
    return(sol)

def d2f1dmdl(theta1, theta2, thetaPoint1, thetaPoint2, params):
    m, l, g = params
    sol = - (thetaPoint2**2 * np.sin(theta1 - theta2))/((1 + m) - m * np.cos(theta1 - theta2)**2) + (m * thetaPoint2**2 * np.sin(theta1 - theta2)*(1 - np.cos(theta1 - theta2)**2))/((1 + m) - m * np.cos(theta1 - theta2)**2)**2
    return(sol)

def d2f1dmdg(theta1, theta2, thetaPoint1, thetaPoint2, params):
    m, l, g = params
    sol = ( -np.sin(theta1) + np.cos(theta1 - theta2)*np.sin(theta2))/((1 + m) - m * np.cos(theta1 - theta2)**2) - ((1 - np.cos(theta1 - theta2)**2) * (m * np.cos(theta1 - theta2) * np.sin(theta2) - (1 + m) * np.sin(theta1)))/((1 + m) - m * np.cos(theta1 - theta2)**2)**2
    return(sol)
    
def d2f2dl2(theta1, theta2, thetaPoint1, thetaPoint2, params):
    m, l, g = params
    f1, f2 = fDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    sol = 2 * thetaPoint1**2 * np.sin(theta1 -theta2)/l**3 - 2 * g * np.sin(theta2)/l**3 - 2 * np.cos(theta1 - theta2)/l**3 * f1 + 2 * np.cos(theta1 - theta2)/l**2 * df1DoublePenduledl(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2)
    return(sol)
    





def rDoublePendule(obs, param, nombre):
    deltat = 30/(nombre-1)
    m, l, g = param
    theta1, theta2 = obs
    thetaPoint1, thetaPoint2 = [0], [0]
    for i in range(nombre-1):
        thetaPoint1.append((theta1[i+1] - theta1[i])/deltat)
        thetaPoint2.append((theta2[i+1] - theta2[i])/deltat)
    sol = []
    for i in range(nombre-1):
        f1, f2 = fDoublePendule(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        sol.append(sqrt((thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat)**2 + (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)**2))
    return(np.array(sol))

def JacrDoublePendule(obs, param, nombre, rinutile):
    deltat = 30/(nombre-1)
    m, l, g = param
    theta1, theta2 = obs
    thetaPoint1, thetaPoint2 = [0], [0]
    for i in range(nombre-1):
        thetaPoint1.append((theta1[i+1] - theta1[i])/deltat)
        thetaPoint2.append((theta2[i+1] - theta2[i])/deltat)
    sol = np.zeros((nombre - 1, 3))
    r = rDoublePendule(obs, param, nombre)
    for i in range(nombre-1):
        f1, f2 = fDoublePendule(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        sol[i,0] = -(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]
        sol[i, 1] = -(deltat*df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]
        sol[i, 2] = -(deltat*df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]
    return(sol)
            
    
def secondTermeHessienneLV(obs, nombre):
    matrice = np.zeros((4,4))
    for i in range(nombre-1):
        matrice = matrice + np.array([[obs[0][i]**2,-obs[0][i]**2*obs[1][i],0,0],[-obs[0][i]**2*obs[1][i], obs[0][i]**2*obs[1][i]**2, 0, 0],[0,0, obs[1][i]**2, -obs[0][i]*obs[1][i]**2],[0,0,-obs[0][i]*obs[1][i]**2, obs[0][i]**2*obs[1][i]**2]])
    deltat = 2/(nombre-1)
    return(matrice*(deltat)**2)

def secondTermeHessienneDoublePendule(obs, nombre, params):
    """  (a + u, b + v, c + w)
         (b + v, 0 + x, 0 + y)
         (c + w, 0 + y, 0 + 0)
         
         la matrice au dessus c'est la matrice des derivés secondes de f, or on veut celle de r, donc je vais devoir la changer un petit peu ?
    """
    
    r = rDoublePendule(obs, params, nombre)
    thetaPoint1, thetaPoint2 = [0], [0]
    theta1, theta2 = obs
    deltat = 30/(nombre-1)
    for i in range(nombre - 1):
        thetaPoint1.append((theta1[i+1] - theta1[i]) / deltat)
        thetaPoint2.append((theta2[i+1] - theta2[i]) / deltat)
    m, l, g = params
    matrice = np.zeros((3,3))
    for i in range(nombre-1):
        a = d2f1dm2(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
        b = d2f1dmdl (theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
        c = d2f1dmdg(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
        
        u = -1/l*np.cos(theta1[i] - theta2[i])*a
        v = 1/l**2 * np.cos(theta1[i] - theta2[i]) * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - 1/l*np.cos(theta1[i] - theta2[i])*b
        w = - 1/l*np.cos(theta1[i] - theta2[i])*c
        x = d2f2dl2(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
        y = np.sin(theta2[i])/l**2 + np.cos(theta1[i] - theta2[i])/l**2 * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        
        f1, f2 = fDoublePendule(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])
        
        matrice = matrice + np.array([[
        deltat / r[i] * (-(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (a * (thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) - deltat * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + u * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
        deltat / r[i] * (-(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (b * (thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) - deltat * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + v * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
        deltat / r[i] * (-(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (c * (thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) - deltat * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + w * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))
        ],[
        deltat / r[i] * (-(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (b * (thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) - deltat * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + v * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
        deltat / r[i] * (-(deltat*df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (- deltat * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + x * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
        deltat / r[i] * (-(deltat*df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (- deltat * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + y * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))
        ],[
        deltat / r[i] * (-(deltat*df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (c * (thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) - deltat * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + w * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
         deltat / r[i] * (-(deltat*df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (- deltat * df1DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledl(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) + y * (thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)),
        deltat / r[i] * (-(deltat*df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + deltat*df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat))/r[i]) * (df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint1[i+1]-thetaPoint1[i] - f1*deltat) + df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])*(thetaPoint2[i+1] - thetaPoint2[i] - f2*deltat)) - deltat * (- deltat * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - deltat * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) * df2DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]))
        ]])
        
    return(matrice)

# def secondTermeHessienneDoublePendule(obs, nombre, params):
#     thetaPoint1, thetaPoint2 = [0], [0]
#     theta1, theta2 = obs
#     deltat = 30/(nombre-1)
#     for i in range(nombre - 1):
#         thetaPoint1.append((theta1[i+1] - theta1[i]) / deltat)
#         thetaPoint2.append((theta2[i+1] - theta2[i]) / deltat)
#     m, l, g = params
#     matrice = np.zeros((3,3))
#     for i in range(nombre):
#         print(i)
#         a = d2f1dm2(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
#         b = d2f1dmdl (theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
#         c = d2f1dmdg(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
#         d = d2f2dl2(theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i], params)
#         matrice = matrice + np.array([[ a + -1/l*np.cos(theta1[i] - theta2[i])*a, b + 1/l**2 * np.cos(theta1[i] - theta2[i]) * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - 1/l*np.cos(theta1[i] - theta2[i])*b, c - 1/l*np.cos(theta1[i] - theta2[i])*c], [b + 1/l**2 * np.cos(theta1[i] - theta2[i]) * df1DoublePenduledm(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]) - 1/l*np.cos(theta1[i] - theta2[i])*b, d, np.sin(theta2[i])/l**2 + np.cos(theta1[i] - theta2[i])/l**2 * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i])], [c - 1/l*np.cos(theta1[i] - theta2[i])*c, np.sin(theta2[i])/l**2 + np.cos(theta1[i] - theta2[i])/l**2 * df1DoublePenduledg(m, l, g, theta1[i], theta2[i], thetaPoint1[i], thetaPoint2[i]), 0]])
#     return(matrice)
    
#II - Méthode de la descente en gradient pour la tester il suffit de lancer "testMethodeDescenteEnGradientLV()"

def methodeDescenteEnGradient(obs, theta0, nombre, pos, titre, gradientS = gradientSDoublePendule, S = SDoublePendule, solutionTheorique = solutionTheoriqueDoublePendule):
    """cette fonction sert à appliquer la méthode de la descente en gradient
    
    ON AJOUTE ICI L'ALGORITHME DE BACKTRACKING LINE SEARCH POUR EXHIBER alpha
    
    """
    theta = theta0                                                              #initialisation de theta
    ancien = 42
    arret = 41

    while arret > 10**(-4):                                                        #Notre condition d'arrêt est quand la norme du gradient de S est inférieur (ou égal) à 0.001 (donc qu'on arrive sur un plateau/partie plate)
        grad = np.array(gradientS(obs, theta, nombre))                             #Si le gradient de S à une dépendance en l'écart-type des observations il faut rajouter cette variable
        
        ancien = arret
        arret = np.dot(grad.T,grad)                                                   #on calcule la norme 1 du gradient pour notre condition d'arrêt
        print(arret)                                                            #permet de suivre la progression du programme
        print(theta)
        alpha = 1
        c, rho = 0.5, 0.9
        
        while  S([theta[i] - alpha*grad[i] for i in range(len(grad))], obs, nombre)  > (S(theta, obs, nombre) - c * alpha * arret):                                                 #cette boucle sert à chercher le meilleur alpha (ie : l'argmax de S(theta - alpha*gradient(S)),
            alpha = alpha * rho
        #print(S([theta[i] - alpha*g[i] for i in range(len(g))], obs, nombre), S(theta, obs, nombre), alpha, theta)
        theta = [theta[i] - alpha*grad[i] for i in range(len(grad))]                  #on applique la formule de récurrence
        
        if ancien == arret:
            break
    
    solutionTheorique(theta[0], theta[1], theta[2], obs[0][0], obs[1][0], 0, 0, pos, titre)               #on trace les résultats
    #solutionTheoriqueDoublePendule(m1, m2, l1, l2, theta1, theta2, g, thetaPoint1, thetaPoint2, pos = [221,223], titre = ["theta 1", "theta 2"])
    
    #cette ligne ne sert qu'à comparer le gradient avec la solution théorique
    grad = np.array(gradientS(obs, [8, 1, 9.80665], nombre))
    print(np.dot(grad.T, grad))
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode


def testMethodeDescenteEnGradientDoublePendule(paramTheorique = [8,1,9.80665], param0 = [7.8,0.9, 10], theta1 = (np.pi)/4, theta2 = (np.pi)/4, thetaPoint1 = 0, thetaPoint2 = 0, ecartTypeBruit = 0.001, nombre = 1001, pos = [[221, 223], [222, 224]], titre = [["theta1 sol théorique", "theta2 sol théorique"], ["theta1 par méthode de la descente en gradient", "theta2 par méthode de la descente en gradient"]]):
    """cette fonction a été pensée pour que notre code assez indépendant du systeme d'equation que l'on cherche à résoudre, et ici on résout le problème de Lotka-Volterra (LV), il y a beaucoup de paramètres mais objectivement peu servent vraiment :
    ecartTypeBruit et nombre servent à générer des observations
    theta0 à avoir une valeur pour initialiser notre algorithme
    thetaTheorique et X0 à pouvoir retracer la solution théorique (donc c'est de la déco)
    pos et titre c'est pour les graphes
    gradientS, S et solutionTheorique c'est pour savoir quel est le système d'équation que l'on utilise
    pas c'est le pas utilisé pour dans la recherche du meilleur alpha dans l'algo
    """
    m, l, g = paramTheorique
    obs = observationDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, ecartTypeBruit, nombre, pos)      #on génère des observations
    solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, pos[0], titre[0])                 #on retrace la courbe théorique (elle est parfois cachée sous les observations)
    return(methodeDescenteEnGradient(obs, param0, nombre, pos[-1], titre[1], gradientSDoublePendule, SDoublePendule, solutionTheoriqueDoublePendule))       #on renvoie le théta trouvé à l'aide de cette méthode
    



#III - Méthode de Gauss-Newton et de Newton pour la tester il suffit de lancer "testMethodesDeNewtonLV()"

def methodeGaussNewton(obs, theta0, nombre, pos, titre, nombreDeRec, solutionTheorique = solutionTheoriqueDoublePendule, rFonction = rDoublePendule, Jacob = JacrDoublePendule):
    """ ici on présente la méthode de Gauss-Newton"""
    theta = theta0                                                              #on initialise theta
    for i in range(nombreDeRec):
        print(i, theta)
        r = rFonction(obs,theta, nombre)                                        #on calcule r
        Jac = Jacob(obs, theta, nombre, r)                                      #on calcule la jacobienne de r
        #print(Jac)
        if np.linalg.det(np.dot(Jac.T, Jac)) == 0.0:
            break
        matrice = np.dot(np.linalg.inv(np.dot(Jac.T, Jac)),Jac.T)
        theta = theta - np.dot(matrice,r.T)                                     #on applique la formule de récurrence
    solutionTheorique(theta[0], theta[1], theta[2], obs[0][0], obs[1][0], 0, 0, pos, titre)               #on trace les résultats
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode

def methodeNewton(obs, theta0, nombre, pos, titre, nombreDeRec, solutionTheorique = solutionTheoriqueDoublePendule, rFonction = rDoublePendule, Jacob = JacrDoublePendule, secondTermeHessienne = secondTermeHessienneDoublePendule):
    """ ici on présente la méthode de Newton sans l'approximation de Gauss-Newton"""
    theta = theta0                                                              #on initialise theta
    matrice = secondTermeHessienne(obs, nombre, theta)                                 #!!!!  le second terme de la hessienne ne dépend que des observations dans notre cas, donc je l'ai mis ici,
                                                                                #!!!!  si ce n'est pas le cas il faut écrire cette ligne dans la boucle for qui suit
    for i in range(nombreDeRec):
        print(i)
        r = rFonction(obs, theta, nombre)                                       #on calcule r
        Jac = Jacob(obs, theta, nombre, r)                                      #on calcule la jacobienne de r
        mat = np.dot(np.linalg.inv(matrice + np.dot(Jac.T, Jac)),Jac.T)         
        theta = theta - np.dot(mat,r.T)                                         #on applique la fomrile de récurrence
    solutionTheorique(theta[0], theta[1], theta[2], obs[0][0], obs[1][0], 0, 0, pos, titre)               #on trace la courbe
    return(theta)                                                               #on renvoie les paramètres trouvés par cette méthode


def testMethodeDeGaussNewtonDoublePendule(paramTheorique = [8,1,9.80665], param0 = [7,0.9, 10], theta1 = (np.pi)/4, theta2 = (np.pi)/4, thetaPoint1 = 0, thetaPoint2 = 0, ecartTypeBruit = 0.001, nombre = 1001, pos = [[231,234], [232, 235], [233, 236]], titre = [["theta1 sol théorique", "theta2 sol théorique"], ["theta1 par méthoded de GN", "theta2 par la méthode de GN"], ["theta1 par Newton", "theta2 par Newton"]], nombreDeRec = 100, solutionTheorique = solutionTheoriqueDoublePendule, rFonction = rDoublePendule, Jacob = JacrDoublePendule, secondTermeHessienne = secondTermeHessienneDoublePendule):
    """cette fonction a été pensée pour que notre code assez indépendant du systeme d'equation que l'on cherche à résoudre :
    ecartTypeBruit et nombre servent à générer des observations
    theta0 à avoir une valeur pour initialiser notre algorithme
    thetaTheorique et X0 à pouvoir retracer la solution théorique (donc c'est de la déco)
    pos et titre c'est pour les graphes
    nombreDeRec correspond au nombre de fois que l'on va appliquer la formule de récurrence
    solutionTheorique, rFonction, Jacob et secondTermeHessienne est lié au système d'équation que l'on étudie
    """
    m, l, g = paramTheorique
    obs = observationDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, ecartTypeBruit, nombre, pos)
    solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, pos[0], titre[0])             #on retrace la courbe théorique (elle est parfois cachée sous les observations)
    a  = methodeGaussNewton(obs, param0, nombre, pos[1], titre[1], nombreDeRec, solutionTheorique, rFonction, Jacob)
    b = methodeNewton(obs, param0, nombre, pos[2], titre[2], nombreDeRec, solutionTheorique, rFonction, Jacob, secondTermeHessienne)        #on applique les deux méthodes, on trace les rendus et on renvoie le paramètre calculé pour chacune de ces méthodes.
    return(a, b)
    
    
    
# IV - La solution boîte noire avec une solution déjà dans python (grâce à scipy.optimize.differential_evolution)

def boiteNoire(obs, nombre, bornes, pos, titre, solutionTheorique, S):
    """ Cette fonction utilise une fonction déjà implémenté dans python pour trouver le paramètre qui minimise S
    """
    sol = differential_evolution(S, bornes, (obs, nombre))                      #on applique la fonction de python qui est la boîte noire
    #print(sol)
    m, l, g = sol.x
    solutionTheorique(m, l, g, obs[0][0], obs[1][0], 0, 0, pos, titre)        #on trace la courbe obtenue
    return(sol)   
    
# def testBoiteNoireLV(ecartTypeBruit = 0.5, nombre = 10001, thetaTheorique = [2,1,4,1], X0 = [5,3], bornes = [(0,5),(0,5),(0,5),(0,5)], pos = [[221, 223], [222, 224]], titre = [["S solution théorique", "W solution théorique"], ["S obtenu avec scipy", "W obtenu avec scipy"]], solutionTheorique = solutionTheoriqueDoublePendule, S = SDoublePendule):
#     """cette fonction sert à tester notre fonction précédente en le testant sur le modèle de Lotka-Volterra
#     """
#     obs = observation(ecartTypeBruit, nombre, pos)
#     solutionTheorique(thetaTheorique, X0, pos[0], titre[0])                     #on trace la solution théorique
#     sol = boiteNoire(obs, nombre, bornes, pos[1], titre[1], solutionTheorique, S)
#     return(sol.x, sol.fun)

def testBoiteNoireDoublePendule(paramTheorique = [8,1,9.80665], bornes = [(7,10),(0,2.5),(8,11)],theta1 = (np.pi)/4, theta2 = (np.pi)/4, thetaPoint1 = 0, thetaPoint2 = 0, ecartTypeBruit = 0, nombre = 1001, pos = [[221, 223], [222, 224]], titre = [["S solution théorique", "W solution théorique"], ["S obtenu avec scipy", "W obtenu avec scipy"]], solutionTheorique = solutionTheoriqueDoublePendule, S = SDoublePendule):
    """cette fonction sert à tester notre fonction précédente en le testant sur le modèle de Lotka-Volterra
    """
    m, l, g = paramTheorique
    
    obs = observationDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, ecartTypeBruit, nombre, pos)
    solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, pos[0], titre[0])        #on trace la solution théorique
    sol = boiteNoire(obs, nombre, bornes, pos[1], titre[1], solutionTheorique, S)
    return(sol.x, sol.fun)
    
    
# V - Si on veut tester les 4 méthodes il suffit de lancer test()

def test(paramTheorique = [8,1,9.80665], param0 = [7,0.9, 10], theta1 = (np.pi)/4, theta2 = (np.pi)/4, thetaPoint1 = 0, thetaPoint2 = 0, ecartTypeBruit = 0.001, nombre = 1001, pos = [[241, 245], [242, 246], [243, 247], [244, 248]], titre = [["S solution théorique","W solution théorique"], ["S par la méthode de descente en gradient", "W par la méthode de descente en gradient"], ["S par la méthode de Gauss-Newton", "W par la méthode de Gauss-Newton"], ["S par la méthode de Newton", "W par la méthode de Newton"], ["S obtenu avec scipy", "W obtenu avec scipy"]], nombreDeRec = 100, gradientS=gradientSDoublePendule, S = SDoublePendule, solutionTheorique = solutionTheoriqueDoublePendule, rFonction = rDoublePendule, Jacob = JacrDoublePendule, secondTermeHessienne = secondTermeHessienneDoublePendule, pas = 10**(-3)):
    """c'est un mélange de tout"""
    m, l, g = paramTheorique
    
    obs = observationDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, ecartTypeBruit, nombre, pos)
    solutionTheoriqueDoublePendule(m, l, g, theta1, theta2, thetaPoint1, thetaPoint2, pos[0], titre[0])
                        
    a = methodeDescenteEnGradient(obs, param0, nombre, pos[1], titre[1], gradientSDoublePendule, SDoublePendule, solutionTheoriqueDoublePendule)
    b  = methodeGaussNewton(obs, param0, nombre, pos[2], titre[2], nombreDeRec, solutionTheorique, rFonction, Jacob)
    c = methodeNewton(obs, param0, nombre, pos[3], titre[3], nombreDeRec, solutionTheorique, rFonction, Jacob, secondTermeHessienne)
    #sol = differential_evolution(S, bornes, (obs, nombre))
    #print(sol)
    #solutionTheorique(sol.x, [obs[i][0] for i in range(len(obs))], pos[4], titre[4])
    return(a,b,c)   
    