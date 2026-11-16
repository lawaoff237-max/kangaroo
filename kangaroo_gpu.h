// =============================================================================
//  kangaroo_gpu.h - interface C plate entre kangaroo.cpp (hote) et kangaroo_gpu.cu
//
//  Le GPU est un pur "generateur de points distingues" : il recoit la table de
//  sauts, l'intervalle et les cles (deja centrees), et renvoie des DPRec bruts.
//  TOUTE la logique de collision (table de hachage partagee, relations k_i=+-k_j+c,
//  sauvegarde de la table tame) reste cote CPU dans kangaroo.cpp, inchangee, et
//  traite les DP venant du GPU exactement comme ceux venant des threads CPU.
//
//  Ce fichier ne depend ni de CUDA ni de la STL : il est inclus tel quel par
//  kangaroo.cpp (compile avec g++) ET par kangaroo_gpu.cu (compile avec nvcc).
// =============================================================================
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Entier 256 bits / element de corps, 4 mots de 64 bits, poids faible en d[0].
// Meme representation que U256/fe cote CPU -> conversion directe champ a champ.
typedef struct { uint64_t d[4]; } gu256_t;

// Un saut de la table (identique a "struct Jump" cote CPU : x,y = point du saut, dist = sa distance).
typedef struct { gu256_t x, y, dist; } gjump_t;

// Une cle publique deja centree (Q_i = pub_i - c*G, comme g_Q cote CPU).
typedef struct { gu256_t x, y; } gpt_t;

// Un point distingue brut, meme contenu que "struct DPRec" cote CPU.
typedef struct {
    uint64_t x0, x1;      // X[0], X[1] du point (mots de poids faible)
    gu256_t  d;           // distance du kangourou a ce point (mod n)
    uint32_t id;          // TAME (0xFFFFFFFF) ou indice de cle (kangourou wild)
    int8_t   s;           // signe +-1 (classe d'equivalence de symetrie)
} gdprec_t;

// Parametres d'initialisation d'un worker GPU = UNE carte physique (gere en
// interne blocks*threads_per_block*kang_per_thread kangourous en parallele).
typedef struct {
    int device_id;            // index CUDA (cudaSetDevice)
    int blocks;                // blocs CUDA
    int threads_per_block;     // threads par bloc
    int kang_per_thread;       // kangourous "logiques" par thread (lot Montgomery interne, cf -gbatch)
    int iters_per_launch;      // iterations internes avant de rendre la main a l'hote (cf -giter)

    int dp_bits;                // masque DP (g_dp), meme convention que le CPU (bits bas a 0)
    int sym;                    // symetrie +-y (g_sym)
    int sl;                     // meme parametre -sl que le CPU (pas L = W >> sl)
    double tfrac;                // fraction tame (g_tfrac)
    int do_tame, do_wild;        // g_doTame / g_doWild au lancement
    uint64_t seed;
    uint64_t rot_limit;          // g_rotLimit cote CPU (0 = pas de rotation forcee des cles wild)

    const gjump_t* jumps;         // NJ sauts normaux suivis de NE sauts d'evasion
    int nj, ne;

    gu256_t secp_n, nhalf;         // ordre de la courbe (g_SECP_N) et floor(n/2)
    gu256_t a, w, c;               // debut, largeur, centre de l'intervalle (g_a, g_W, g_c)

    const gpt_t* keys;              // Q_i centres (g_Q), nk entrees
    int nk;

    int out_capacity;               // taille du tampon de DP alloue par appel a gpu_worker_run
} gpu_init_t;

// gpu_device_count() : renvoie 0 si ce binaire est compile sans CUDA (voir les
// bouchons dans kangaroo.cpp) ou si aucune carte n'est detectee a l'execution.
int  gpu_device_count(void);
void gpu_device_name(int device_id, char* buf, int buflen);

// Cree et initialise un worker pour la carte device_id (copie la table de sauts,
// les constantes de courbe et les cles en memoire GPU). Renvoie NULL en cas d'echec
// (carte absente, memoire insuffisante, etc. - le message d'erreur va sur stderr).
void* gpu_worker_create(const gpu_init_t* cfg);

// A appeler avant CHAQUE gpu_worker_run si l'etat a change cote hote :
//   solved[i] != 0  -> la cle i est resolue (les wild ne doivent plus la choisir)
//   tame_off   != 0 -> mode -mix : les kangourous tame repartent tous en wild
void gpu_worker_sync(void* h, const uint8_t* solved, int nk, int tame_off);

// Fait avancer tous les kangourous de ce worker de iters_per_launch iterations.
// Remplit out[] (jusqu'a out_capacity entrees) des points distingues trouves durant
// l'appel et renvoie leur nombre (peut etre tronque si plus de DP que out_capacity
// sont trouves pendant cet appel : rare si out_capacity est raisonnable, cf -giter).
// *steps et *tame_steps recoivent le nombre total de sauts (resp. sauts "tame")
// effectues par ce worker durant l'appel (pour les statistiques affichees).
int gpu_worker_run(void* h, gdprec_t* out, int out_capacity, uint64_t* steps, uint64_t* tame_steps);

void gpu_worker_destroy(void* h);

#ifdef __cplusplus
}
#endif
