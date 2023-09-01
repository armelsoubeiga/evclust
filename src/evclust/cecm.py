# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains the main function for cecm : Contraints ecm
"""

#---------------------- Packges------------------------------------------------
#from evclust.utils import makeF, extractMass, setCentersECM, setDistances, solqp
import numpy as np
from scipy.sparse import coo_matrix, tril


def cecm(x, c, type='full', pairs=None, ntrials=1, ML=None, CL=None, g0=None, alpha=1,
         delta=10, xi=0.5, distance=0, epsi=1e-3, disp=True):
    """
    Constrained Evidential c-means algorithm.

    cecm computes a credal partition from a matrix of attribute data and pairwise constraints using the Constrained
    Evidential c-means (CECM) algorithm.

    CECM is a version of ECM allowing the user to specify pairwise constraints to guide the clustering process.
    Pairwise constraints are of two kinds: must-link constraints are pairs of objects that are known to belong to the
    same class, and cannot-link constraints are pairs of objects that are known to belong to different classes. CECM can
    also learn a metric for each cluster, like the Gustafson-Kessel algorithm in fuzzy clustering. At each iteration,
    the algorithm solves a quadratic programming problem using an interior ellipsoidal trust region and barrier function
    algorithm with dual solution updating technique in the standard QP form (Ye, 1992).

    If initial prototypes g0 are provided, the number of trials is automatically set to 1.

    Parameters:
    - x: Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
    - c: Number of clusters.
    - type: Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega; "pairs":
      empty set, singletons, Omega, and all or selected pairs).
    - pairs: Set of pairs to be included in the focal sets. If None, all pairs are included. Used only if type="pairs".
    - ntrials: Number of runs of the optimization algorithm (set to 1 if g0 is supplied).
    - ML: Matrix nbML x 2 of must-link constraints. Each row of ML contains the indices of objects that belong to the same
      class.
    - CL: Matrix nbCL x 2 of cannot-link constraints. Each row of CL contains the indices of objects that belong to
      different classes.
    - g0: Initial prototypes, matrix of size c x d. If not supplied, the prototypes are initialized randomly.
    - alpha: Exponent of the cardinality in the cost function.
    - delta: Distance to the empty set.
    - xi: Tradeoff between the objective function Jecm and the constraints: Jcecm=(1-xi)Jecm + xi Jconst.
    - distance: Type of distance use: 0=Euclidean, 1=Mahalanobis.
    - epsi: Minimum amount of improvement.
    - disp: If True, intermediate results are displayed.

    Returns:
    The credal partition (an object of class "credpart").

    References:
    - V. Antoine, B. Quost, M.-H. Masson and T. Denoeux. CECM: Constrained Evidential C-Means algorithm.
      Computational Statistics and Data Analysis, Vol. 56, Issue 4, pages 894-914, 2012.
    - Y. Ye. On affine-scaling algorithm for nonconvex quadratic programming. Math. Programming 56 (1992) 285-300.

    Author: Thierry Denoeux (from a MATLAB code written by Violaine Antoine).
    """

    bal = xi
    x = np.asmatrix(x)
    n = x.shape[0]
    nbAtt = x.shape[1]
    beta = 2
    K = c
    rho2 = delta**2
    F = makeF(K, type, pairs)
    nbFoc = F.shape[0]
    card = np.sum(F[1:nbFoc, :], axis=1)
    
    if ntrials > 1 and g0 is not None:
        print('WARNING: ntrials>1 and g0 provided. Parameter ntrials set to 1.')
        ntrials = 1
    
    # --------------- constraint matrix reformulation --------------
    nbML = ML.shape[0]
    contraintesML = coo_matrix((np.ones(nbML), (ML[:, 0], ML[:, 1])), shape=(n, n))
    contraintesML.setdiag(0)
    contraintesML = contraintesML + contraintesML.T
    contraintesML = contraintesML.sign()
    contraintesML = contraintesML * tril(contraintesML)
    nbML = contraintesML.nnz
    
    nbCL = CL.shape[0]
    contraintesCL = coo_matrix((np.ones(nbCL), (CL[:, 0], CL[:, 1])), shape=(n, n))
    contraintesCL.setdiag(0)
    contraintesCL = contraintesCL + contraintesCL.T
    contraintesCL = contraintesCL.sign()
    contraintesCL = contraintesCL * tril(contraintesCL)
    nbCL = contraintesCL.nnz
    
    # -- Setting q vector  
    nbContParObjet = np.sum(contraintesML + contraintesML.T, axis=1)
    q = np.kron(nbContParObjet, np.array([[1] + [0] * (nbFoc - 1)])).flatten()
    q = q.astype(float)
    
    # -- Setting constraints matrix
    if nbML == 0:
        nbML = 1
    if nbCL == 0:
        nbCL = 1
    
    MLMat = np.asarray(np.sum(F, axis=1) == 1, dtype=np.float64)
    MLMat = np.dot(MLMat.reshape(-1, 1), MLMat.reshape(1, -1)) * np.dot(F, F.T)
    CLMat = np.sign(np.dot(F, F.T))
    MLMat = MLMat * -np.sign(bal) / (2 * nbML)
    CLMat = CLMat * np.sign(bal) / (2 * nbCL)
    
    #MLMatTiled = np.kron(np.eye(n), MLMat)
    #CLMatTiled = np.kron(np.eye(n), CLMat)
    MLaux = np.kron(np.eye(n), MLMat)
    CLaux = np.kron(np.eye(n), CLMat)
    
    #contraintesMat = np.block([[MLMatTiled, np.zeros((n * nbFoc, nbCL * nbFoc))],[np.zeros((n * nbCL, nbML * nbFoc)), CLMatTiled]])
    contraintesMat = np.kron(np.ones((n,n)), MLMat) * MLaux + np.kron(np.ones((n,n)), CLMat) * CLaux
    contraintesMat = contraintesMat + contraintesMat.T
    
    Jbest = np.inf
    for itrial in range(ntrials):
        # ------------------- initializations ------------------------
        if g0 is None:
            g = x[np.random.choice(n, K), :] + 0.1 * np.random.randn(K * nbAtt).reshape(K, nbAtt)
        else:
            g = g0
        
        gplus = np.zeros((nbFoc - 1, nbAtt))
        for i in range(1, nbFoc):
            fi = F[i, :]
            truc = np.dot(np.ones((K, 1)), fi.reshape(1, nbAtt))
            gplus[i-1, :] = np.sum(g * truc, axis=0) / np.sum(fi)
        
        D = np.zeros((n, nbFoc - 1))
        for j in range(nbFoc - 1):
            D[:, j] = np.sum((np.array(x) - np.tile(gplus[j, :], (n, 1)))**2, axis=1)
        
        m = np.zeros((n, nbFoc - 1))
        for i in range(n):
            vect0 = D[i, :]
            for j in range(nbFoc - 1):
                vect1 = (np.tile(D[i, j], nbFoc - 1) / vect0) ** (1 / (beta - 1))
                vect2 = np.tile(card[j] ** (alpha / (beta - 1)), nbFoc - 1) / (card ** (alpha / (beta - 1)))
                vect3 = vect1 * vect2
                m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / rho2) ** (1 / (beta - 1)))
        
        m1 = np.hstack((1 - np.sum(m, axis=1).reshape(-1, 1), m))
        
        dis = setDistances(x, F, g, m, alpha, distance)
        D = dis['D']
        Smeans = dis['Smean']
        
        # -- Setting H matrix
        aux = np.dot(D, np.hstack((np.zeros((nbFoc - 1, 1)), np.eye(nbFoc - 1)))) + np.hstack((np.ones((n, 1)) * rho2, np.zeros((n, nbFoc - 1))))
        vectDist = np.ravel(aux.T)
        
        Card = np.concatenate(([1], card))
        Card = np.tile(Card ** alpha, n)

        H = (1 - bal) * np.diag(vectDist * Card / (n * nbFoc)) + bal * contraintesMat

        # --------------------- iterations ---------------------------
        notfinished = True
        gold = g
        iter = 0
        Aeq = np.kron(np.eye(n), np.ones((1, nbFoc)))
        #Aeq = Aeq.T
        beq = np.ones(n)
        
        while notfinished:
            iter += 1
            mvec0 = m1.ravel()  # masses used as initialization
            
            qp = solqp(Q=H, A=Aeq, b=beq, c=q, x=mvec0)
            masses = qp['x']
            m1 = masses.reshape(n, nbFoc)
            m = m1[:, 1:nbFoc]
            
            g = setCentersECM(x, m, F, Smeans, alpha, beta)
            dist = setDistances(x, F, g, m, alpha, distance)
            D = dist['D']
            Smeans = dist['Smean']
            
            # H matrix
            aux = np.dot(D, np.hstack((np.zeros((nbFoc - 1, 1)), np.eye(nbFoc - 1)))) + np.hstack((np.ones((n, 1)) * rho2, np.zeros((n, nbFoc - 1))))
            vectDist = np.ravel(aux.T)
            H = (1 - bal) * np.diag(vectDist * Card / (n * nbFoc)) + bal * contraintesMat

            
            J = np.dot(np.dot(masses.T, H), masses) + bal
            delta = np.max(np.abs(g - gold))
            if disp:
                print([iter, J, delta])
            
            notfinished = delta > epsi
            gold = g
        
        if J < Jbest:
            Jbest = J
            mbest = m1
            gbest = g
            Smeansbest = Smeans
        
        res = [itrial, J, Jbest]
        if disp:
            print(res)
    
    clus = extractMass(mbest, F, g=gbest, S=Smeansbest, method="cecm", crit=Jbest,
                       param=dict(alpha=alpha, beta=beta, delta=delta))
    return clus
