# =============================================================================
#  Makefile - kangaroo (secp256k1), CPU multi-coeurs + GPU CUDA optionnel
#
#  "make"       : detecte nvcc automatiquement.
#                  - trouve  -> binaire CPU+GPU (kangaroo_gpu.cu compile avec nvcc)
#                  - absent  -> binaire CPU seul, identique au programme original
#  "make cpu"   : force un binaire CPU seul, meme si nvcc est installe.
#  "make kaggle": raccourci pense pour les notebooks Kaggle (2x T4, sm_75).
#  "make clean" : nettoie les fichiers objets et le binaire.
#
#  Le lien final passe TOUJOURS par nvcc (jamais g++ directement) : nvcc connait
#  deja le bon chemin vers libcudart pour le toolkit installe, ce qui evite de le
#  deviner (source d'un "cannot find -lcudart" selon l'image/la distribution).
# =============================================================================
CXX      ?= g++
NVCC     := $(shell command -v nvcc 2>/dev/null)
CXXFLAGS ?= -O3 -march=native -flto -funroll-loops -std=c++17 -pthread
NVCCFLAGS?= -O3 -std=c++17 --extended-lambda -diag-suppress 20091 -Xcompiler "-O3 -pthread"
# sm_75 = Turing (T4, Kaggle). Ajoutez d'autres architectures si besoin, ex: sm_80 (A100).
GENCODE  ?= -gencode arch=compute_75,code=sm_75

BIN = kangaroo

.PHONY: all cpu kaggle clean
all:
ifeq ($(NVCC),)
	@echo "nvcc introuvable : build CPU seul (identique a l'original)."
	$(CXX) $(CXXFLAGS) kangaroo.cpp -o $(BIN)
else
	@echo "nvcc trouve ($(NVCC)) : build CPU + GPU."
	$(NVCC) $(NVCCFLAGS) $(GENCODE) -c kangaroo_gpu.cu -o kangaroo_gpu.o
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA -c kangaroo.cpp -o kangaroo.o
	$(NVCC) $(NVCCFLAGS) kangaroo.o kangaroo_gpu.o -o $(BIN)
endif

cpu:
	$(CXX) $(CXXFLAGS) kangaroo.cpp -o $(BIN)

# Kaggle : 2x Nvidia T4 (Turing, sm_75), CUDA deja installe dans l'image standard.
kaggle:
	nvcc $(NVCCFLAGS) $(GENCODE) -c kangaroo_gpu.cu -o kangaroo_gpu.o
	g++ $(CXXFLAGS) -DHAVE_CUDA -c kangaroo.cpp -o kangaroo.o
	nvcc $(NVCCFLAGS) kangaroo.o kangaroo_gpu.o -o $(BIN)
	@echo "OK : ./kangaroo -gpu all  (utilise les 2 T4 + tous les coeurs CPU)"

clean:
	rm -f $(BIN) kangaroo.o kangaroo_gpu.o
