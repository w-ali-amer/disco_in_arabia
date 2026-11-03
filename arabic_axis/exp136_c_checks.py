import sys; sys.path.insert(0,".")
import numpy as np, exp136_span as S, exp136_rung2 as R
rng=np.random.default_rng(21)
def proj(d,rank):
    if rank==0: return np.zeros((d,d),complex)
    Z=rng.standard_normal((d,rank))+1j*rng.standard_normal((d,rank)); V,_=np.linalg.qr(Z); return V@V.conj().T
def rho_set(d):
    out={}
    for x in R.X:
        v=rng.standard_normal(d)+1j*rng.standard_normal(d); v/=np.linalg.norm(v); out[x]=np.outer(v,v.conj())
    return out
W=R.words("Cb"); n=len(W); iu=np.triu_indices(n)
def cb_vec(d,ranks):
    M=[proj(d,ranks()) for _ in range(6)]; return R.moment(W,rho_set(d),M,d).real[iu]
for name,rk2,rk8 in (("rank1",lambda:1,lambda:1),("union012",lambda:int(rng.integers(3)),None)):
    for d in (2,8):
        if d==8 and rk8 is None:   # union at d=8: ranks 0..8
            f=lambda: cb_vec(8,lambda:int(rng.integers(9)))
        else:
            f=lambda: cb_vec(d,rk2 if d==2 else rk8)
        Q,_,_=R.grow(f,n*(n+1)//2); print("Cb",name,"d",d,"real span",Q.shape[1],flush=True)
# S3 d=4 vs 16, first two effects of fixed ranks (third outcome dropped)
for pat in ((1,1),(1,2),(2,1)):
    for d in (4,16):
        for level in S.LEVELS:
            def f():
                rho,_=S.sample(rng,d,3,3); eff=[]
                for _ in range(3):
                    Z=rng.standard_normal((d,sum(pat)))+1j*rng.standard_normal((d,sum(pat))); V,_=np.linalg.qr(Z)
                    eff+= [V[:,:pat[0]]@V[:,:pat[0]].conj().T, V[:,pat[0]:]@V[:,pat[0]:].conj().T]
                Wd=S.words(rho,eff,level,d); A=np.array([w.reshape(-1) for w in Wd]); G=A.conj()@A.T; i=np.triu_indices(len(Wd)); return G[i].real
            nn={"L1":13,"L1+rhoM":49,"L1+rhoM+MM":85}[level]; Q,_,_=R.grow(f,nn*(nn+1)//2); print("S3 proj",pat,level,"d",d,"real span",Q.shape[1],flush=True)
