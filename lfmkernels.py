#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:23:58 2017

@author: guarni
"""
import numpy as np
from scipy.special import wofz

def kern_ode(t, tp, C, B, S, lq):
    C = np.asarray([C,])
    B = np.asarray([B,])
    index = np.zeros((np.size(t),1),dtype=np.int)
    indexp = np.zeros((np.size(tp),1),dtype=np.int)
    def hnew(gam1, gam2, gam3, t, tp, wofznu, nu, nu2):
        c1_1 = 1./(gam2 + gam1)
        c1_2 = 1./(gam3 + gam1)
        
        tp_lq = tp/lq
        t_lq = t/lq
        dif_t_lq = t_lq - tp_lq
        #Exponentials        
        #egam1tp = np.exp(-gam1*tp)
        gam1tp = gam1*tp
        egam2t = np.exp(-gam2*t)
        egam3t = np.exp(-gam3*t)
        #squared exponentials        
        #edif_t_lq = np.exp(-dif_t_lq*dif_t_lq)
        dif_t_lq2 = dif_t_lq*dif_t_lq
        #et_lq = np.exp(-t_lq*t_lq)
        t_lq2 = t_lq*t_lq
        #etp_lq = np.exp(-tp_lq*tp_lq)
        tp_lq2 = tp_lq*tp_lq
        ec = egam2t*c1_1 - egam3t*c1_2
    
        #Terms of h
        A = dif_t_lq + nu
        temp = np.zeros(A.shape, dtype = complex)
        boolT = A.real >=0.
        if np.any(boolT):
            wofzA = wofz(1j*A[boolT])
            temp[boolT] = np.exp(np.log(wofzA) - dif_t_lq2[boolT])
        boolT = np.logical_not(boolT)
        if np.any(boolT):
            dif_t = t[boolT] - tp[boolT]
            wofzA = wofz(-1j*A[boolT])
            temp[boolT] = 2.*np.exp(nu2[boolT] + gam1[boolT]*dif_t) - np.exp(np.log(wofzA) - dif_t_lq2[boolT])

        b = t_lq + nu
        wofzb = wofz(1j*b)
        e = nu - tp_lq
        temp2 = np.zeros(e.shape, dtype = complex)
        boolT = e.real >= 0.
        if np.any(boolT):
            wofze = wofz(1j*e[boolT])
            temp2[boolT] = np.exp(np.log(wofze) - tp_lq2[boolT])
        boolT = np.logical_not(boolT)
        if np.any(boolT):
            wofze = wofz(-1j*e[boolT])
            temp2[boolT] = 2.*np.exp(nu2[boolT] - gam1tp[boolT]) - np.exp(np.log(wofze) - tp_lq2[boolT])
        return (c1_1 - c1_2)*(temp \
            - np.exp(np.log(wofzb) - t_lq2 - gam1tp))\
            - ec*( temp2 \
            - np.exp(np.log(wofznu) - gam1tp))
    
    indexp = indexp.reshape((1,indexp.size))
    alpha = .5*C
    w = .5*np.sqrt(4.*B - C*C + 0j)
    wbool = C*C>4.*B
    wbool = np.logical_or(wbool[:,None],wbool)

    ind2t, ind2tp = np.where(wbool[index,indexp])
    ind3t, ind3tp = np.where(np.logical_not(wbool[index,indexp])) #TODO: from the original index can be done

    gam = alpha + 1j*w
    gamc = alpha - 1j*w
    W = w*w.reshape((w.size,1))
    K0 = S**2*lq*np.sqrt(np.pi)/(8.*W[index, indexp])
    nu = lq*gam/2.
    nu2 = nu*nu
    wofznu = wofz(1j*nu)
    
    kff = np.zeros((t.size, tp.size), dtype = complex)

    t = t.reshape(t.size,)
    tp = tp.reshape(tp.size,)    
    index = index.reshape(index.size,)
    indexp = indexp.reshape(indexp.size,)
    indbf, indbc = np.where(np.ones(kff.shape, dtype=bool))
    index2 = index[indbf]
    index2p = indexp[indbc]
    #Common computation for both cases
    kff[indbf,indbc] = hnew(gam[index2p], gamc[index2], gam[index2], t[indbf], tp[indbc], wofznu[index2p], nu[index2p], nu2[index2p])\
       + hnew(gam[index2], gamc[index2p], gam[index2p], tp[indbc], t[indbf], wofznu[index2], nu[index2], nu2[index2])
    
    #Now we calculate when w_d or w_d' are not real
    if np.any(wbool):
        #Precomputations
        nuc = lq*gamc/2.
        nuc2 = nuc**2
        wofznuc = wofz(1j*nuc)
        #A new way to work with indexes
        ind = index[ind2t]
        indp = indexp[ind2tp]
        t1 = t[ind2t]
        t2 = tp[ind2tp]
        kff[ind2t, ind2tp] += hnew(gamc[indp], gam[ind], gamc[ind], t1, t2, wofznuc[indp], nuc[indp], nuc2[indp])\
         + hnew(gamc[ind], gam[indp], gamc[indp], t2, t1, wofznuc[ind], nuc[ind], nuc2[ind])
        
    #When wd and wd' ares real
    if np.any(np.logical_not(wbool)):
        kff[ind3t, ind3tp] = 2.*np.real(kff[ind3t, ind3tp])

    return (K0 * kff).real

def kern_ode_Kuu(t, lq):
    index = np.zeros((np.size(t),1),dtype=np.int)
    lq = np.asarray([lq,])
    
    index = index.reshape(index.size,)
    t = t[:, 0].reshape(t.shape[0],)
    lq = lq.reshape(lq.size,)
    lq2 = lq*lq
    #Covariance matrix initialization
    kuu = np.zeros((t.size, t.size))
    #Assign 1. to diagonal terms
    kuu[np.diag_indices(t.size)] = 1.
    #Upper triangular indices
    indtri1, indtri2 = np.triu_indices(t.size, 1)
    #Block Diagonal indices among Upper Triangular indices
    ind = np.where(index[indtri1] == index[indtri2])
    indr = indtri1[ind]
    indc = indtri2[ind]
    r = t[indr] - t[indc]
    r2 = r*r
    #Calculation of  covariance function
    kuu[indr, indc] = np.exp(-r2/lq2[index[indr]])
    #Completation of lower triangular part
    kuu[indc, indr] = kuu[indr, indc]
    return kuu

def kern_ode_Kfu(t, tp, C, B, S, lq):
    C = np.asarray([C,])
    B = np.asarray([B,])
    lq = np.asarray([lq,])
    S = np.asarray([S,])
    
    lq = lq.reshape(lq.size,)
    S = S.reshape(S.size,)
    index = np.zeros((np.size(t),),dtype=np.int)
    index2 = np.zeros((np.size(tp),),dtype=np.int)
    
    t = t[:, 0].reshape(t.shape[0], 1)
    d = np.unique(index) #Output Indexes
    B = B[d]
    C = C[d]
    #Index transformation
    indd = np.arange(1)
    indd[d] = np.arange(d.size)
    index = indd[index]
    #Check where wd becomes complex
    wbool = C*C >= 4.*B
    #Output related variables must be column-wise
    C = C.reshape(C.size, 1)
    B = B.reshape(B.size, 1)
    C2 = C*C
    #Input related variables must be row-wise
    z = tp[:, 0].reshape(1, tp.shape[0])
    lq = lq.reshape((1, lq.size))
    #print np.max(z), np.max(z/lq[0, index2])
    alpha = .5*C

    wbool2 = wbool[index]
    ind2t = np.where(wbool2)
    ind3t = np.where(np.logical_not(wbool2))

    kfu = np.empty((t.size, z.size))

    indD = np.arange(B.size)
    #(1) when wd is real
    if np.any(np.logical_not(wbool)):
        #Indexes of index and t related to (2)
        t1 = t[ind3t]
        ind = index[ind3t]
        #Index transformation
        d = np.asarray(np.where(np.logical_not(wbool))[0])
        indd = indD.copy()
        indd[d] = np.arange(d.size)
        ind = indd[ind]
        #Dx1 terms
        w = .5*np.sqrt(4.*B[d] - C2[d])
        alphad = alpha[d]
        gam = alphad - 1j*w

        #DxQ terms
        Slq = (S[d]/w)*(.5*lq)
        c0 = Slq*np.sqrt(np.pi)
        nu = gam*(.5*lq)
        #1xM terms
        z_lq = z/lq[0, index2]
        #NxQ terms
        t_lq = t1/lq
        #NxM terms
        zt_lq = z_lq - t_lq[:, index2]

        # Upsilon Calculations
        #Using wofz
        tz = t1-z
        fullind = np.ix_(ind, index2)
        zt_lq2 = -zt_lq*zt_lq
        z_lq2 = -z_lq*z_lq
        gamt = -gam[ind]*t1

        upsi = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind]))))
        z1 = zt_lq + nu[fullind]
        indv1 = np.where(z1.real >= 0.)
        indv2 = np.where(z1.real < 0.)
        if indv1[0].size > 0:
            upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
        if indv2[0].size > 0:
            nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
            upsi[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                           - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
        upsi[t1[:, 0] == 0., :] = 0.

        #Covariance calculation
        kfu[ind3t] = c0[fullind]*upsi.imag

    #(2) when wd is complex
    if np.any(wbool):
        #Indexes of index and t related to (2)
        t1 = t[ind2t]
        ind = index[ind2t]
        #Index transformation
        d = np.asarray(np.where(wbool)[0])
        indd = indD.copy()
        indd[d] = np.arange(d.size)
        ind = indd[ind]
        #Dx1 terms
        w = .5*np.sqrt(C2[d] - 4.*B[d])
        alphad = alpha[d]
        gam = alphad - w
        gamc = alphad + w
        #DxQ terms
        Slq = S[d]*(lq*.25)
        c0 = -Slq*(np.sqrt(np.pi)/w)
        nu = gam*(lq*.5)
        nuc = gamc*(lq*.5)
        #1xM terms
        z_lq = z/lq[0, index2]
        #NxQ terms
        t_lq = t1/lq[0, index2]
        #NxM terms
        zt_lq = z_lq - t_lq

        # Upsilon Calculations
        tz = t1-z
        z_lq2 = -z_lq*z_lq
        zt_lq2 = -zt_lq*zt_lq
        gamt = -gam[ind]*t1
        gamct = -gamc[ind]*t1
        fullind = np.ix_(ind, index2)
        upsi = np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind])).real))\
               - np.exp(z_lq2 + gamct + np.log(wofz(1j*(z_lq + nuc[fullind])).real))

        z1 = zt_lq + nu[fullind]
        indv1 = np.where(z1 >= 0.)
        indv2 = np.where(z1 < 0.)
        if indv1[0].size > 0:
            upsi[indv1] -= np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
        if indv2[0].size > 0:
            nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
            upsi[indv2] -= np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                           - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
        z1 = zt_lq + nuc[fullind]
        indv1 = np.where(z1 >= 0.)
        indv2 = np.where(z1 < 0.)
        if indv1[0].size > 0:
            upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
        if indv2[0].size > 0:
            nuac2 = nuc[ind[indv2[0]], index2[indv2[1]]]**2
            upsi[indv2] += np.exp(nuac2 - gamc[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                           - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
        upsi[t1[:, 0] == 0., :] = 0.

        kfu[ind2t] = c0[np.ix_(ind, index2)]*upsi
    return kfu