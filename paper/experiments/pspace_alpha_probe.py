"""Remark (alpha): negative-side decay rate of the warped profile vs wrap-around
(8x8 acoustic sponge, p_max=8, n_p=9). Out: pspace_alpha_probe_results.json"""
import sys, numpy as np
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from multiprocessing import get_context
from scipy.sparse.linalg import expm_multiply
from lib import WaveEquationSponge2d, Schrodingerisation

def gprof(p, alpha, delta):
    g = np.where(p>=0, np.exp(-p), np.exp(-alpha*np.abs(p)))
    m = (p>-delta)&(p<0)
    # cubic Hermite on [-delta,0]: g(-d)=e^{-a d}, g'(-d)=a e^{-a d}, g(0)=1, g'(0)=-1
    y0, d0 = np.exp(-alpha*delta), alpha*np.exp(-alpha*delta)
    y1, d1 = 1.0, -1.0
    t = (p[m]+delta)/delta
    h00=2*t**3-3*t**2+1; h10=t**3-2*t**2+t; h01=-2*t**3+3*t**2; h11=t**3-t**2
    g[m] = h00*y0 + h10*delta*d0 + h01*y1 + h11*delta*d1
    return g

def job(a):
    alpha, delta, sm, T, n_p, pmax = a
    N=8; b=np.exp(-0.5*((np.arange(N)-4)/1.0)**2)
    op=WaveEquationSponge2d(3,3,npml=2,sigma_max=sm); z0=op.initial_state(np.outer(b,b))
    sch=Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=n_p, p_max=pmax)
    if alpha is not None: sch.g = gprof(sch.p, alpha, delta)
    W=sch.evolve(z0,T); zT=expm_multiply(op.generator().tocsc()*T,z0)
    z,ps=sch.recover(W)
    return alpha, delta, sm, T, np.linalg.norm(z-zT)/np.linalg.norm(z0), np.linalg.norm(z-zT)/np.linalg.norm(zT)

jobs=[(al,de,1.0,T,9,8.0) for (al,de) in [(None,None),(1,1.0),(4,1.0),(8,1.0),(16,0.5)] for T in (2,4,6,8,10,12,14)]
if __name__ == '__main__':
    with get_context('fork').Pool(8) as pool:
        res = pool.map(job, jobs)
    for r in res: print(r[0],r[1],r[2],r[3],'abs %.2e rel %.2e'%(r[4],r[5]))
    import json
    json.dump([dict(alpha=r[0], delta=r[1], sigma_max=r[2], T=r[3], abs_err=float(r[4]), rel_err=float(r[5])) for r in res],
              open(os.path.join(HERE, 'pspace_alpha_probe_results.json'), 'w'), indent=1)
