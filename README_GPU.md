# kangaroo - CPU + GPU (CUDA, pensé pour 1-2x Nvidia T4 / Kaggle)

## Ce qui a changé par rapport à l'original

Le fichier `kangaroo.cpp` d'origine (CPU, multi-coeurs) est **conservé intact dans sa
logique** : arithmétique 256 bits, résolution des collisions (`add_dp`/`collide`/
`propagate`/`try_key`), table des points distingués (DP), sauvegarde/chargement de la
table tame. Rien de tout cela n'a bougé.

Ce qui est ajouté :
- `kangaroo_gpu.h` : interface C plate entre l'hôte et le GPU (types simples, pas de C++/STL).
- `kangaroo_gpu.cu` : le code CUDA lui-même (compilé uniquement par `nvcc`).
- Quelques options CLI (`-gpu`, `-gblocks`, `-gthreads`, `-gbatch`, `-giter`) et le
  branchement dans `main()` de `kangaroo.cpp` (détection des cartes, lancement d'un
  thread "pont" par carte).

**Principe clé** : le GPU ne fait QUE produire des points distingués, à très haut débit
(des centaines de milliers à quelques millions de kangourous en vol par carte). Il
renvoie ces DP bruts au CPU, qui les insère dans les **mêmes** tables de hachage
partagées que les threads CPU, via `add_dp()` (inchangée, protégée par mutex). Toute la
logique de résolution (trouver la clé, gérer les relations k_i = ±k_j + c, sauvegarder
la table tame) reste donc strictement identique entre CPU et GPU, et les deux
travaillent sur le **même** problème en parallèle sans aucun risque d'incohérence.

## Compilation

```bash
make            # detecte nvcc automatiquement (GPU+CPU si present, sinon CPU seul)
make cpu        # force un binaire CPU seul (comportement 100% original)
make kaggle     # raccourci identique a "make", cible explicitement sm_75 (T4)
```

Sans `nvcc` installé, `kangaroo.cpp` compile et se comporte **exactement** comme le
fichier original : `gpu_device_count()` renvoie 0 et aucun thread GPU n'est créé (voir
les bouchons `#ifndef HAVE_CUDA` en tête de fichier).

## Sur Kaggle (2x T4)

1. Notebook -> Settings -> Accelerator -> GPU T4 x2.
2. Dans une cellule :
   ```bash
   !nvidia-smi          # verifie que les 2 T4 sont visibles
   !nvcc --version       # CUDA deja installe sur les images Kaggle
   !make kaggle
   ```
3. Lancer une recherche en utilisant CPU + les 2 GPU en même temps :
   ```bash
   ./kangaroo -pub <cle_publique_hex> -bb 66 -gpu all -c 4
   ```
   (`-c 4` = 4 threads CPU en plus des 2 GPU ; `-gpu off` désactive le GPU, `-gpu 0`
   n'utilise que la première carte.)

## Nouvelles options

| Option        | Rôle                                                                 | Défaut |
|---------------|-----------------------------------------------------------------------|--------|
| `-gpu <spec>` | `all` (toutes les cartes), `off`, ou liste `0,1`                      | `all`  |
| `-gblocks n`  | blocs CUDA par carte                                                    | 1024   |
| `-gthreads n` | threads par bloc                                                        | 128    |
| `-gbatch n`   | kangourous "logiques" par thread (lot Montgomery interne, max 32)       | 16     |
| `-giter n`    | itérations internes par lancement de noyau avant de rendre la main      | 64     |
| `-c 0`        | désactive les threads CPU (GPU seul)                                   | -      |

Nombre de kangourous en vol par carte ≈ `gblocks * gthreads * gbatch` (par défaut :
1024×128×16 ≈ 2,1 millions). À ajuster selon la mémoire de la carte si vous montez ces
valeurs (chaque kangourou utilise environ 250 octets de mémoire GPU persistante).

## Limites connues de ce portage GPU (honnêteté avant tout)

Je n'ai pas pu compiler ni exécuter ce code sur une vraie carte CUDA dans cet
environnement (pas de GPU ici). J'ai donc :
- vérifié indépendamment l'arithmétique 256 bits portée (inversion, 2G, addition
  affine du saut) avec des vecteurs de test connus de secp256k1 ;
- vérifié la syntaxe et le typage de l'intégralité de `kangaroo_gpu.cu` en le
  compilant avec `g++` via des bouchons qui neutralisent juste les mots-clés propres à
  CUDA (`__global__`, `<<<...>>>`, etc.) - ça ne remplace pas un vrai test sur GPU, mais
  ça élimine déjà fautes de frappe, incohérences de types et paramètres.

Avant un grand calcul, je recommande de lancer d'abord un test court et résoluble
(comme dans l'exemple ci-dessous) pour vérifier que le GPU retrouve bien la clé, avant
de lancer une recherche longue :
```bash
PUB=$(./kangaroo -genpub 1a2b3c4d5e)
./kangaroo -pub $PUB -bb 40 -gpu all -c 0 -t 5
```

Simplifications assumées par rapport au CPU (cas rares, qui n'affectent pas la
correction du résultat final mais peuvent légèrement réduire l'efficacité) :
- fenêtre d'historique de détection de cycle identique (16 valeurs) mais testée par
  kangourou individuellement comme côté CPU - pas de simplification réelle ici ;
- si le tampon de DP d'un lancement de noyau est saturé (par défaut 262 144 entrées par
  carte et par appel), les DP excédentaires de cet appel sont perdus (pas un bug, juste
  une perte d'efficacité) - cf `-giter` pour réduire le débit par appel si besoin sur de
  tout petits intervalles de test où `-dp` est choisi très bas ;
- `-end`/`-mix` sont pris en compte GPU (bascule tame/wild), mais la granularité de la
  transition est celle d'un lancement de noyau (quelques dizaines de millions de sauts),
  pas instantanée comme sur CPU.

## Nécessite CUDA ≥ 11.5

Le code utilise `unsigned __int128` en code `__device__` pour l'arithmétique 256 bits
(comme le fait le CPU), supporté par nvcc depuis la version 11.5 avec un compilateur
hôte gcc/clang (jamais MSVC). Les images Kaggle standard embarquent une version de CUDA
bien plus récente (12.x), donc aucune action nécessaire de votre part.
