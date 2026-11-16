// =============================================================================
//  kangaroo_gpu.cu  -  backend CUDA du kangourou de Pollard (secp256k1)
//
//  Principe : chaque thread CUDA fait avancer un petit lot (GB = -gbatch,
//  defaut 16) de kangourous "logiques", avec EXACTEMENT le meme algorithme que
//  worker() cote CPU (jump table, inversion Montgomery par lot, symetrie +-y,
//  detection de cycle court, points distingues). Le GPU ne fait QUE produire des
//  DP bruts : la resolution (collisions, relations, table tame, sauvegarde) reste
//  intacte dans kangaroo.cpp et traite les DP du GPU comme ceux du CPU.
//
//  Compile uniquement avec nvcc (voir le Makefile) ; jamais avec g++.
//  Necessite CUDA >= 11.5 pour le support de unsigned __int128 en code __device__
//  (present sur les images Kaggle standard, cf nvcc --version).
// =============================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include "kangaroo_gpu.h"
using std::min; using std::max;

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

#define HD __host__ __device__ __forceinline__
#define NJ 256
#define NE 64
#define HBUF 16
#define MAXGB 32
#define TAME 0xFFFFFFFFu

// même représentation que gu256_t : 4 mots 64 bits, poids faible en d[0]
typedef gu256_t U256;
typedef gu256_t fe;

#define P0 0xFFFFFFFEFFFFFC2FULL
#define CP 0x1000003D1ULL

// ============================ arithmetique 256 bits (HD) =====================
// Copie fidele des fonctions du fichier CPU (kangaroo.cpp), juste qualifiees
// __host__ __device__. Verifiee independamment (2G, inversion vs pow) avant
// integration - voir le commentaire en fin de fichier.
HD int clz64(u64 x) {
#if defined(__CUDA_ARCH__)
    return x ? __clzll((long long)x) : 64;
#else
    return x ? __builtin_clzll(x) : 64;
#endif
}
HD bool u_zero(const U256& a) { return !(a.d[0] | a.d[1] | a.d[2] | a.d[3]); }
HD bool u_eq(const U256& a, const U256& b) { return a.d[0]==b.d[0]&&a.d[1]==b.d[1]&&a.d[2]==b.d[2]&&a.d[3]==b.d[3]; }
HD int u_cmp(const U256& a, const U256& b) { for (int i=3;i>=0;i--) if (a.d[i]!=b.d[i]) return a.d[i]>b.d[i]?1:-1; return 0; }
HD U256 u_zero_val() { U256 r; r.d[0]=r.d[1]=r.d[2]=r.d[3]=0; return r; }
HD U256 u_from(u64 v) { U256 r; r.d[0]=v; r.d[1]=r.d[2]=r.d[3]=0; return r; }
HD U256 u_add(const U256& a, const U256& b) { U256 r; u128 c=0; for (int i=0;i<4;i++){ c+=(u128)a.d[i]+b.d[i]; r.d[i]=(u64)c; c>>=64; } return r; }
HD U256 u_sub(const U256& a, const U256& b) { U256 r; u64 br=0; for (int i=0;i<4;i++){ u128 t=(u128)a.d[i]-b.d[i]-br; r.d[i]=(u64)t; br=(u64)(t>>64)&1; } return r; }
HD int u_bits(const U256& a) { for (int i=3;i>=0;i--) if (a.d[i]) return i*64+64-clz64(a.d[i]); return 0; }
HD void add_small(U256& r, u64 v) { u128 c=(u128)r.d[0]+v; r.d[0]=(u64)c; c>>=64; for (int i=1;i<4&&c;i++){ c+=r.d[i]; r.d[i]=(u64)c; c>>=64; } }
HD void sub_small(U256& r, u64 v) { u64 br=r.d[0]<v; r.d[0]-=v; for (int i=1;i<4&&br;i++){ br=(r.d[i]==0); r.d[i]--; } }
HD U256 u_shl1(const U256& a) { U256 r; for (int i=3;i>0;i--) r.d[i]=(a.d[i]<<1)|(a.d[i-1]>>63); r.d[0]=a.d[0]<<1; return r; }

__device__ __constant__ U256 c_secp_n, c_nhalf, c_a, c_w, c_c;

HD U256 m_add(const U256& a, const U256& b, const U256& n) {
    U256 r; u128 c=0; for (int i=0;i<4;i++){ c+=(u128)a.d[i]+b.d[i]; r.d[i]=(u64)c; c>>=64; }
    if (c || u_cmp(r,n)>=0) r = u_sub(r,n);
    return r;
}
HD U256 m_sub(const U256& a, const U256& b, const U256& n) {
    U256 r; u64 br=0; for (int i=0;i<4;i++){ u128 t=(u128)a.d[i]-b.d[i]-br; r.d[i]=(u64)t; br=(u64)(t>>64)&1; }
    if (br) r = u_add(r,n);
    return r;
}
HD U256 m_neg(const U256& a, const U256& n) { return u_zero(a) ? a : u_sub(n,a); }
HD U256 m_sgn(const U256& a, int s, const U256& n) { return s<0 ? m_neg(a,n) : a; }

HD void fe_norm(fe& r) { if (r.d[3]==~0ULL && r.d[2]==~0ULL && r.d[1]==~0ULL && r.d[0]>=P0) add_small(r,CP); }
HD void fe_add(fe& r, const fe& a, const fe& b) {
    u128 c=0; for (int i=0;i<4;i++){ c+=(u128)a.d[i]+b.d[i]; r.d[i]=(u64)c; c>>=64; }
    if (c) add_small(r,CP);
    fe_norm(r);
}
HD void fe_sub(fe& r, const fe& a, const fe& b) {
    u64 br=0; for (int i=0;i<4;i++){ u128 t=(u128)a.d[i]-b.d[i]-br; r.d[i]=(u64)t; br=(u64)(t>>64)&1; }
    if (br) sub_small(r,CP);
}
HD void fe_neg(fe& r, const fe& a) { if (u_zero(a)) { r=a; return; } fe z=u_zero_val(); fe_sub(r,z,a); }
HD void fe_reduce_from_t(fe& r, u64 t[8]) {
    u64 o[4]; u128 c=0;
    for (int i=0;i<4;i++){ c += (u128)t[4+i]*CP + t[i]; o[i]=(u64)c; c>>=64; }
    u64 top=(u64)c; c=(u128)top*CP;
    for (int i=0;i<4;i++){ c+=o[i]; o[i]=(u64)c; c>>=64; }
    r.d[0]=o[0]; r.d[1]=o[1]; r.d[2]=o[2]; r.d[3]=o[3];
    if (c) add_small(r,CP);
    fe_norm(r);
}
HD void fe_mul(fe& r, const fe& a, const fe& b) {
    u64 t[8]; u128 c=0;
    for (int j=0;j<4;j++){ c += (u128)a.d[0]*b.d[j]; t[j]=(u64)c; c>>=64; } t[4]=(u64)c;
    for (int i=1;i<4;i++){
        c=0;
        for (int j=0;j<4;j++){ c += (u128)a.d[i]*b.d[j]+t[i+j]; t[i+j]=(u64)c; c>>=64; }
        t[i+4]=(u64)c;
    }
    fe_reduce_from_t(r,t);
}
HD void fe_sqr(fe& r, const fe& a) { fe_mul(r,a,a); }
HD void fe_sqrn(fe& r, const fe& a, int n) { r=a; for (int i=0;i<n;i++) fe_sqr(r,r); }
HD void fe_inv(fe& r, const fe& a) {
    fe x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqrn(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqrn(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqrn(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqrn(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqrn(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqrn(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqrn(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqrn(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqrn(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqrn(t,x223,23); fe_mul(t,t,x22);
    fe_sqrn(t,t,5); fe_mul(t,t,a);
    fe_sqrn(t,t,3); fe_mul(t,t,x2);
    fe_sqrn(t,t,2); fe_mul(r,t,a);
}
// ---- courbe (Jacobian, utilisee seulement a l'init/reinit d'un kangourou) ----
struct Jac { fe X,Y,Z; };
HD Jac jac_inf() { Jac r; r.X=u_from(1); r.Y=u_from(1); r.Z=u_zero_val(); return r; }
HD void jac_dbl(Jac& r, const Jac& a) {
    if (u_zero(a.Z) || u_zero(a.Y)) { r = jac_inf(); return; }
    fe A,B,C,D,E,F,t,X3,Y3,Z3,C8;
    fe_sqr(A,a.X); fe_sqr(B,a.Y); fe_sqr(C,B);
    fe_add(t,a.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A); fe_sqr(F,E);
    fe_sub(X3,F,D); fe_sub(X3,X3,D);
    fe_sub(t,D,X3); fe_mul(Y3,E,t);
    fe_add(C8,C,C); fe_add(C8,C8,C8); fe_add(C8,C8,C8);
    fe_sub(Y3,Y3,C8);
    fe_mul(Z3,a.Y,a.Z); fe_add(Z3,Z3,Z3);
    r.X=X3; r.Y=Y3; r.Z=Z3;
}
HD void jac_add(Jac& r, const Jac& a, const Jac& b) {
    if (u_zero(a.Z)) { r=b; return; } if (u_zero(b.Z)) { r=a; return; }
    fe Z1Z1,Z2Z2,U1,U2,S1,S2,H,R,t;
    fe_sqr(Z1Z1,a.Z); fe_sqr(Z2Z2,b.Z);
    fe_mul(U1,a.X,Z2Z2); fe_mul(U2,b.X,Z1Z1);
    fe_mul(t,b.Z,Z2Z2); fe_mul(S1,a.Y,t);
    fe_mul(t,a.Z,Z1Z1); fe_mul(S2,b.Y,t);
    fe_sub(H,U2,U1); fe_sub(R,S2,S1);
    if (u_zero(H)) { if (u_zero(R)) { jac_dbl(r,a); return; } r=jac_inf(); return; }
    fe H2,H3,V,X3,Y3,Z3;
    fe_sqr(H2,H); fe_mul(H3,H,H2); fe_mul(V,U1,H2);
    fe_sqr(X3,R); fe_sub(X3,X3,H3); fe_sub(X3,X3,V); fe_sub(X3,X3,V);
    fe_sub(t,V,X3); fe_mul(Y3,R,t); fe_mul(t,S1,H3); fe_sub(Y3,Y3,t);
    fe_mul(Z3,a.Z,b.Z); fe_mul(Z3,Z3,H);
    r.X=X3; r.Y=Y3; r.Z=Z3;
}
struct GPt { fe x,y; bool inf; };
HD GPt to_aff(const Jac& j) {
    GPt p; p.inf=false;
    if (u_zero(j.Z)) { p.inf=true; p.x=u_zero_val(); p.y=u_zero_val(); return p; }
    fe zi,z2,z3; fe_inv(zi,j.Z); fe_sqr(z2,zi); fe_mul(z3,z2,zi);
    fe_mul(p.x,j.X,z2); fe_mul(p.y,j.Y,z3);
    return p;
}
HD Jac jac_from(const GPt& p) { if (p.inf) return jac_inf(); Jac r; r.X=p.x; r.Y=p.y; r.Z=u_from(1); return r; }
HD GPt pt_mul(const U256& k, const GPt& base) {
    Jac R=jac_inf(), B=jac_from(base);
    int nb=u_bits(k);
    for (int i=nb-1;i>=0;i--) { jac_dbl(R,R); if ((k.d[i/64]>>(i%64))&1) jac_add(R,R,B); }
    return to_aff(R);
}
HD GPt pt_add(const GPt& a, const GPt& b) { Jac r; jac_add(r, jac_from(a), jac_from(b)); return to_aff(r); }
HD GPt pt_neg(const GPt& p) { GPt r=p; if (!p.inf) fe_neg(r.y,p.y); return r; }

__device__ __constant__ U256 c_GX, c_GY;
HD GPt GEN_pt() { GPt g; g.x=c_GX; g.y=c_GY; g.inf=false; return g; }
HD GPt mulG(const U256& k) { return pt_mul(k, GEN_pt()); }
HD GPt mulG_mod(const U256& k) {
    if (u_cmp(k, c_nhalf) > 0) return pt_neg(mulG(u_sub(c_secp_n, k)));
    return mulG(k);
}

// ---- generateur xoshiro256** (copie du Rng cote CPU) ----
HD u64 rotl64(u64 x, int k) { return (x << k) | (x >> (64-k)); }
HD void rng_seed(u64 s[4], u64 seed) {
    for (int i=0;i<4;i++) {
        seed += 0x9E3779B97F4A7C15ULL; u64 z=seed;
        z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
        s[i]=z^(z>>31);
    }
}
HD u64 rng_next(u64 s[4]) {
    u64 r=rotl64(s[1]*5,7)*9, t=s[1]<<17;
    s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl64(s[3],45);
    return r;
}
HD U256 u_ones(int k) { U256 r=u_zero_val(); for (int i=0;i<4;i++){ int lo=i*64; if (k>=lo+64) r.d[i]=~0ULL; else if (k>lo) r.d[i]=(1ULL<<(k-lo))-1; } return r; }
HD U256 rng_below(u64 s[4], const U256& n) {
    if (u_zero(n)) return u_zero_val();
    U256 m1=n; sub_small(m1,1);
    U256 mask = u_ones(u_bits(m1));
    for (;;) {
        U256 r; for (int i=0;i<4;i++) r.d[i] = rng_next(s) & mask.d[i];
        if (u_cmp(r,n) < 0) return r;
    }
}

// =============================== etat GPU =====================================
// Table de sauts et cles en memoire constante/globale GPU (copiees une fois a la creation).
__device__ __constant__ gjump_t c_jumps[NJ+NE];
__device__ __constant__ int c_sym, c_dp_bits, c_sl, c_do_tame, c_do_wild, c_nk, c_tame_off;
__device__ __constant__ double c_tfrac;
__device__ __constant__ u64 c_dpmask;
__device__ __constant__ u64 c_rotLimit;    // rotation forcee des kangourous wild (0 = desactivee)

struct KangState {
    fe *X, *Y; U256 *D; u32 *ID; int8_t *S; u32 *SN, *HP, *AGE; u64 *H; u64 *rng;
};

HD bool roleTame(int j, int GB) {
    if (!c_do_tame || c_tame_off) return false;
    if (!c_do_wild) return true;
    double tf = c_tfrac;
    return (long)floor((j+1)*tf) > (long)floor(j*tf);
}

// pioche une cle wild non resolue (round-robin global, comme pick_key() cote CPU)
__device__ u64 d_keyctr;
__device__ int device_pick_key(const uint8_t* solved, int nk) {
    if (nk <= 0) return -1;
    for (int t=0;t<nk;t++) {
        u64 v = atomicAdd((unsigned long long*)&d_keyctr, 1ULL);
        int i = (int)(v % (u64)nk);
        if (!solved[i]) return i;
    }
    return -1;
}

// initialise (ou reinitialise) le kangourou logique j du thread courant.
// renvoie false si aucune cle n'est disponible (toutes resolues) : le kangourou
// est alors "gare" (S[*]=0, ignore par les phases 1/2/3 jusqu'au prochain appel).
__device__ bool init_one(u64 rs[4], fe& X, fe& Y, U256& D, u32& ID, int8_t& S,
                          u32& SN, u32& HP, u32& AGE, u64 H[HBUF], int j, int GB,
                          const gpt_t* keys, const uint8_t* solved, int nk) {
    bool tame = roleTame(j, GB);
    U256 L;
    if (c_sl >= 255) L = u_from(1);
    else {
        int n = c_sl, w = n >> 6, b = n & 63;
        U256 r = u_zero_val();
        for (int i=0;i+w<4;i++) { u64 v = c_w.d[i+w] >> b; if (b && i+w+1<4) v |= c_w.d[i+w+1] << (64-b); r.d[i]=v; }
        L = r;
    }
    if (u_zero(L)) L = u_from(1);
    U256 L2p1 = u_shl1(L); add_small(L2p1,1);
    for (int tries=0; tries<64; tries++) {
        U256 x = rng_below(rs, L2p1);
        U256 r = (u_cmp(x,L) >= 0) ? u_sub(x,L) : m_neg(u_sub(L,x), c_secp_n);
        if (u_zero(r)) r = u_from(1);
        GPt P; int id=-1;
        if (tame) P = mulG_mod(r);
        else {
            id = device_pick_key(solved, nk);
            if (id < 0) { S = 0; return false; }
            GPt Qi; Qi.x = keys[id].x; Qi.y = keys[id].y; Qi.inf=false;
            P = pt_add(Qi, mulG_mod(r));
        }
        if (P.inf) continue;
        X=P.x; Y=P.y; D=r; ID = tame ? TAME : (u32)id; S=1;
        if (c_sym && (Y.d[0] & 1)) { fe_neg(Y,Y); D = m_neg(D, c_secp_n); S=-1; }
        for (int h=0;h<HBUF;h++) H[h]=0;
        HP=0; H[HP++ & (HBUF-1)] = X.d[0];
        SN=0; AGE=0;
        return true;
    }
    S = 0; return false;   // intervalle degenere : abandonne ce kangourou pour cet appel
}

// addition lente (dx==0 : doublement ou P=-J), identique a slow_add() cote CPU.
__device__ const gjump_t* slow_add(fe& x, fe& y, const gjump_t* jp) {
    fe lam,t,x3,y3,inv;
    if (u_eq(x,jp->x)) {
        if (u_eq(y,jp->y)) {
            fe n,d; fe_sqr(n,x); fe_add(t,n,n); fe_add(n,t,n);
            fe_add(d,y,y); fe_inv(inv,d); fe_mul(lam,n,inv);
        } else {
            jp = &c_jumps[NJ];               // saut d'evasion n.0 (cf g_E[0] cote CPU)
            fe dx; fe_sub(dx,jp->x,x); fe_inv(inv,dx);
            fe_sub(t,jp->y,y); fe_mul(lam,t,inv);
        }
    } else {
        fe dx; fe_sub(dx,jp->x,x); fe_inv(inv,dx);
        fe_sub(t,jp->y,y); fe_mul(lam,t,inv);
    }
    fe l2; fe_sqr(l2,lam);
    fe_sub(x3,l2,x); fe_sub(x3,x3,jp->x);
    fe_sub(t,x,x3); fe_mul(y3,lam,t); fe_sub(y3,y3,y);
    x=x3; y=y3;
    return jp;
}

// =============================== noyau principal ==============================
__global__ void kang_kernel(KangState st, int N, int GB, int iters,
                             const gpt_t* keys, const uint8_t* solved, int nk,
                             gdprec_t* outBuf, int outCap, unsigned int* outCount,
                             unsigned long long* stepsAccum, unsigned long long* tameAccum,
                             u32 dpLimit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    long base = (long)tid * GB;

    u64 rs[4]; for (int k=0;k<4;k++) rs[k] = st.rng[tid*4+k];
    fe X[MAXGB], Y[MAXGB], DX[MAXGB], PRE[MAXGB];
    U256 D[MAXGB]; u32 ID[MAXGB]; int8_t S[MAXGB];
    u32 SN[MAXGB], HP[MAXGB], AGE[MAXGB];
    u64 H[MAXGB][HBUF];
    uint8_t SLOW[MAXGB];
    const gjump_t* JP[MAXGB];

    for (int j=0;j<GB;j++) {
        X[j]=st.X[base+j]; Y[j]=st.Y[base+j]; D[j]=st.D[base+j]; ID[j]=st.ID[base+j]; S[j]=st.S[base+j];
        SN[j]=st.SN[base+j]; HP[j]=st.HP[base+j]; AGE[j]=st.AGE[base+j];
        for (int h=0;h<HBUF;h++) H[j][h]=st.H[(base+j)*HBUF+h];
        if (S[j]==0) {       // kangourou "gare" (pas de cle dispo au dernier appel) : on retente
            init_one(rs, X[j], Y[j], D[j], ID[j], S[j], SN[j], HP[j], AGE[j], H[j], j, GB, keys, solved, nk);
        }
    }

    u64 localSteps=0, localTame=0;
    for (int it=0; it<iters; it++) {
        // phase 1 : choix du saut
        for (int j=0;j<GB;j++) {
            if (S[j]==0) { DX[j]=u_from(1); SLOW[j]=1; JP[j]=&c_jumps[0]; continue; }
            const gjump_t* jp = nullptr;
            if (c_sym) {
                u32 pos = HP[j]; u64 xl = X[j].d[0];
                for (int Lc=2; Lc<=8; Lc+=2) {
                    if (H[j][(pos-1-Lc)&(HBUF-1)] == xl) {
                        u64 mn=xl; for (int q=1;q<Lc;q++){ u64 v=H[j][(pos-1-q)&(HBUF-1)]; if (v<mn) mn=v; }
                        jp = &c_jumps[NJ + (int)((mn>>7) % NE)];
                        for (int h=0;h<HBUF;h++) H[j][h]=0;
                        HP[j]=0; H[j][HP[j]++ & (HBUF-1)] = xl;
                        break;
                    }
                }
            }
            if (!jp) jp = &c_jumps[X[j].d[1] & (NJ-1)];
            JP[j] = jp;
            fe_sub(DX[j], jp->x, X[j]);
            if (u_zero(DX[j])) { SLOW[j]=1; DX[j]=u_from(1); } else SLOW[j]=0;
        }
        // phase 2 : inversion par lot (Montgomery), serielle sur le petit lot GB du thread
        PRE[0]=DX[0];
        for (int j=1;j<GB;j++) fe_mul(PRE[j], PRE[j-1], DX[j]);
        fe inv; fe_inv(inv, PRE[GB-1]);
        for (int j=GB-1;j>0;j--) { fe t; fe_mul(t, inv, PRE[j-1]); fe_mul(inv, inv, DX[j]); DX[j]=t; }
        DX[0]=inv;
        // phase 3 : addition affine + MAJ distance/symetrie + DP
        u64 tameStep=0;
        for (int j=0;j<GB;j++) {
            if (S[j]==0) continue;
            const gjump_t* jp = JP[j];
            if (SLOW[j]) jp = slow_add(X[j], Y[j], jp);
            else {
                fe t,lam,l2,x3,y3;
                fe_sub(t, jp->y, Y[j]); fe_mul(lam, t, DX[j]);
                fe_sqr(l2, lam);
                fe_sub(x3, l2, X[j]); fe_sub(x3, x3, jp->x);
                fe_sub(t, X[j], x3); fe_mul(y3, lam, t); fe_sub(y3, y3, Y[j]);
                X[j]=x3; Y[j]=y3;
            }
            D[j] = m_add(D[j], jp->dist, c_secp_n);
            if (c_sym && (Y[j].d[0] & 1)) { fe_neg(Y[j], Y[j]); D[j] = m_neg(D[j], c_secp_n); S[j] = -S[j]; }
            H[j][HP[j]++ & (HBUF-1)] = X[j].d[0];
            SN[j]++; AGE[j]++;
            if (ID[j] == TAME) tameStep++;
            if ((X[j].d[0] & c_dpmask) == 0) {
                unsigned int slot = atomicAdd(outCount, 1u);
                if (slot < (unsigned)outCap) {
                    gdprec_t r; r.x0=X[j].d[0]; r.x1=X[j].d[1]; r.d=D[j]; r.id=ID[j]; r.s=S[j];
                    outBuf[slot] = r;
                }
                SN[j]=0;
            } else if (SN[j] > dpLimit) {
                init_one(rs, X[j], Y[j], D[j], ID[j], S[j], SN[j], HP[j], AGE[j], H[j], j, GB, keys, solved, nk);
            }
        }
        localSteps += GB; localTame += tameStep;
        if ((it & 63) == 0) {
            for (int j=0;j<GB;j++) {
                if (S[j]==0) continue;
                if (ID[j]==TAME) { if (c_tame_off) init_one(rs,X[j],Y[j],D[j],ID[j],S[j],SN[j],HP[j],AGE[j],H[j],j,GB,keys,solved,nk); }
                else if ((ID[j] < (u32)nk && solved[ID[j]]) || (c_rotLimit && AGE[j] > c_rotLimit))
                    init_one(rs,X[j],Y[j],D[j],ID[j],S[j],SN[j],HP[j],AGE[j],H[j],j,GB,keys,solved,nk);
            }
        }
    }

    for (int j=0;j<GB;j++) {
        st.X[base+j]=X[j]; st.Y[base+j]=Y[j]; st.D[base+j]=D[j]; st.ID[base+j]=ID[j]; st.S[base+j]=S[j];
        st.SN[base+j]=SN[j]; st.HP[base+j]=HP[j]; st.AGE[base+j]=AGE[j];
        for (int h=0;h<HBUF;h++) st.H[(base+j)*HBUF+h]=H[j][h];
    }
    for (int k=0;k<4;k++) st.rng[tid*4+k]=rs[k];
    if (localSteps) atomicAdd(stepsAccum, (unsigned long long)localSteps);
    if (localTame)  atomicAdd(tameAccum,  (unsigned long long)localTame);
}

__global__ void init_kernel(KangState st, int N, int GB, const gpt_t* keys, const uint8_t* solved, int nk, u64 seedBase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    u64 rs[4]; rng_seed(rs, seedBase + (u64)tid * 0x9E3779B97F4A7C15ULL + 7919ULL);
    long base = (long)tid * GB;
    for (int j=0;j<GB;j++) {
        fe X,Y; U256 D; u32 ID; int8_t S; u32 SN,HP,AGE; u64 H[HBUF];
        init_one(rs, X, Y, D, ID, S, SN, HP, AGE, H, j, GB, keys, solved, nk);
        st.X[base+j]=X; st.Y[base+j]=Y; st.D[base+j]=D; st.ID[base+j]=ID; st.S[base+j]=S;
        st.SN[base+j]=SN; st.HP[base+j]=HP; st.AGE[base+j]=AGE;
        for (int h=0;h<HBUF;h++) st.H[(base+j)*HBUF+h]=H[h];
    }
    for (int k=0;k<4;k++) st.rng[tid*4+k]=rs[k];
}

// ============================ cote hote (API C) ================================
struct GpuHandle {
    int device;
    int N, GB, iters, outCap;
    KangState st;
    gpt_t* d_keys; uint8_t* d_solved; int nk;
    gdprec_t* d_out; unsigned int* d_outCount;
    unsigned long long *d_steps, *d_tame;
    u32 dpLimit;
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA erreur (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return nullptr; } } while (0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA erreur (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); } } while (0)

extern "C" int gpu_device_count(void) {
    int n=0; cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess) ? n : 0;
}
extern "C" void gpu_device_name(int device_id, char* buf, int buflen) {
    if (!buf || buflen<=0) return;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, device_id) == cudaSuccess)
        snprintf(buf, buflen, "%s (%d SM, %.1f Go)", p.name, p.multiProcessorCount, p.totalGlobalMem/1073741824.0);
    else snprintf(buf, buflen, "GPU %d", device_id);
}

extern "C" void* gpu_worker_create(const gpu_init_t* cfg) {
    if (cfg->nj != NJ || cfg->ne != NE) {
        fprintf(stderr, "kangaroo_gpu: table de sauts incompatible (nj=%d ne=%d attendu %d/%d)\n", cfg->nj, cfg->ne, NJ, NE);
        return nullptr;
    }
    if (cfg->kang_per_thread < 1 || cfg->kang_per_thread > MAXGB) {
        fprintf(stderr, "kangaroo_gpu: -gbatch doit etre entre 1 et %d\n", MAXGB);
        return nullptr;
    }
    CK(cudaSetDevice(cfg->device_id));
    GpuHandle* h = new GpuHandle();
    h->device = cfg->device_id;
    h->N = cfg->blocks * cfg->threads_per_block;
    h->GB = cfg->kang_per_thread;
    h->iters = max(1, cfg->iters_per_launch);
    h->outCap = max(1, cfg->out_capacity);
    long total = (long)h->N * h->GB;
    h->nk = cfg->nk;

    CK(cudaMalloc(&h->st.X, total*sizeof(fe)));   CK(cudaMalloc(&h->st.Y, total*sizeof(fe)));
    CK(cudaMalloc(&h->st.D, total*sizeof(U256))); CK(cudaMalloc(&h->st.ID, total*sizeof(u32)));
    CK(cudaMalloc(&h->st.S, total*sizeof(int8_t)));
    CK(cudaMalloc(&h->st.SN, total*sizeof(u32))); CK(cudaMalloc(&h->st.HP, total*sizeof(u32)));
    CK(cudaMalloc(&h->st.AGE, total*sizeof(u32)));
    CK(cudaMalloc(&h->st.H, total*HBUF*sizeof(u64)));
    CK(cudaMalloc(&h->st.rng, (long)h->N*4*sizeof(u64)));

    CK(cudaMalloc(&h->d_keys, max(1,cfg->nk)*sizeof(gpt_t)));
    if (cfg->nk > 0) CK(cudaMemcpy(h->d_keys, cfg->keys, cfg->nk*sizeof(gpt_t), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&h->d_solved, max(1,cfg->nk)*sizeof(uint8_t)));
    CK(cudaMemset(h->d_solved, 0, max(1,cfg->nk)*sizeof(uint8_t)));

    CK(cudaMalloc(&h->d_out, max(1,cfg->out_capacity)*sizeof(gdprec_t)));
    CK(cudaMalloc(&h->d_outCount, sizeof(unsigned int)));
    CK(cudaMalloc(&h->d_steps, sizeof(unsigned long long)));
    CK(cudaMalloc(&h->d_tame, sizeof(unsigned long long)));

    // constantes de courbe / intervalle
    U256 GX, GY;  // generateur secp256k1, en dur (identique a init_curve() cote CPU)
    { const char* gx="79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
      const char* gy="483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";
      auto hex2u=[&](const char* s, U256& r){ r=u_zero_val(); int len=(int)strlen(s), pos=0;
          for (int i=len-1;i>=0;i--,pos++){ char c=s[i]; int v= (c<='9')?c-'0':(c<='F'?c-'A'+10:c-'a'+10);
              r.d[pos/16] |= (u64)v << ((pos%16)*4); } };
      hex2u(gx, GX); hex2u(gy, GY);
    }
    CK(cudaMemcpyToSymbol(c_GX, &GX, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_GY, &GY, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_secp_n, &cfg->secp_n, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_nhalf, &cfg->nhalf, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_a, &cfg->a, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_w, &cfg->w, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_c, &cfg->c, sizeof(U256)));
    CK(cudaMemcpyToSymbol(c_jumps, cfg->jumps, (NJ+NE)*sizeof(gjump_t)));
    int sym=cfg->sym, dpb=cfg->dp_bits, sl=cfg->sl, dt=cfg->do_tame, dw=cfg->do_wild, nk=cfg->nk, toff=0;
    double tf=cfg->tfrac;
    u64 dpmask = (dpb>=64) ? ~0ULL : ((1ULL<<dpb)-1);
    CK(cudaMemcpyToSymbol(c_sym,&sym,sizeof(int)));       CK(cudaMemcpyToSymbol(c_dp_bits,&dpb,sizeof(int)));
    CK(cudaMemcpyToSymbol(c_sl,&sl,sizeof(int)));         CK(cudaMemcpyToSymbol(c_do_tame,&dt,sizeof(int)));
    CK(cudaMemcpyToSymbol(c_do_wild,&dw,sizeof(int)));    CK(cudaMemcpyToSymbol(c_nk,&nk,sizeof(int)));
    CK(cudaMemcpyToSymbol(c_tame_off,&toff,sizeof(int))); CK(cudaMemcpyToSymbol(c_tfrac,&tf,sizeof(double)));
    CK(cudaMemcpyToSymbol(c_dpmask,&dpmask,sizeof(u64)));
    CK(cudaMemcpyToSymbol(c_rotLimit,&cfg->rot_limit,sizeof(u64)));
    h->dpLimit = (u32)min((u64)0xFFFFFFF0ULL, ((u64)64 << min(dpb,24)) + 4096);
    u64 kc0 = 0; CK(cudaMemcpyToSymbol(d_keyctr, &kc0, sizeof(u64)));

    dim3 blk(cfg->threads_per_block), grid(cfg->blocks);
    init_kernel<<<grid, blk>>>(h->st, h->N, h->GB, h->d_keys, h->d_solved, h->nk, cfg->seed);
    CK(cudaDeviceSynchronize());
    return h;
}

extern "C" void gpu_worker_sync(void* hv, const uint8_t* solved, int nk, int tame_off) {
    GpuHandle* h = (GpuHandle*)hv;
    cudaSetDevice(h->device);
    if (nk > 0) CKV(cudaMemcpy(h->d_solved, solved, nk*sizeof(uint8_t), cudaMemcpyHostToDevice));
    CKV(cudaMemcpyToSymbol(c_tame_off, &tame_off, sizeof(int)));
}

extern "C" int gpu_worker_run(void* hv, gdprec_t* out, int out_capacity, uint64_t* steps, uint64_t* tame_steps) {
    GpuHandle* h = (GpuHandle*)hv;
    cudaSetDevice(h->device);
    unsigned int zero=0; unsigned long long zero64=0;
    CKV(cudaMemcpy(h->d_outCount, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CKV(cudaMemcpy(h->d_steps, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CKV(cudaMemcpy(h->d_tame, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice));

    int threads = 128; int blocks = (h->N + threads - 1) / threads;
    // (threads/blocks de lancement = decoupage technique des N threads logiques deja
    //  alloues a la creation ; independant de -gblocks/-gthreads d'origine)
    int capThisCall = min(out_capacity, h->outCap);
    kang_kernel<<<blocks, threads>>>(h->st, h->N, h->GB, h->iters,
                                      h->d_keys, h->d_solved, h->nk,
                                      h->d_out, capThisCall, h->d_outCount,
                                      h->d_steps, h->d_tame, h->dpLimit);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "kangaroo_gpu: erreur noyau: %s\n", cudaGetErrorString(e)); return 0; }

    unsigned int cnt=0; CKV(cudaMemcpy(&cnt, h->d_outCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int n = (int)min(cnt, (unsigned)capThisCall);
    if (n > 0) CKV(cudaMemcpy(out, h->d_out, n*sizeof(gdprec_t), cudaMemcpyDeviceToHost));
    unsigned long long s=0,t=0;
    CKV(cudaMemcpy(&s, h->d_steps, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CKV(cudaMemcpy(&t, h->d_tame, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (steps) *steps = s;
    if (tame_steps) *tame_steps = t;
    return n;
}

extern "C" void gpu_worker_destroy(void* hv) {
    GpuHandle* h = (GpuHandle*)hv;
    if (!h) return;
    cudaSetDevice(h->device);
    cudaFree(h->st.X); cudaFree(h->st.Y); cudaFree(h->st.D); cudaFree(h->st.ID); cudaFree(h->st.S);
    cudaFree(h->st.SN); cudaFree(h->st.HP); cudaFree(h->st.AGE); cudaFree(h->st.H); cudaFree(h->st.rng);
    cudaFree(h->d_keys); cudaFree(h->d_solved); cudaFree(h->d_out); cudaFree(h->d_outCount);
    cudaFree(h->d_steps); cudaFree(h->d_tame);
    delete h;
}
