# =============================================================================
#  Makefile - kangaroo (secp256k1), CPU multi-coeurs + GPU CUDA optionnel
#
#  "make"       : detecte nvcc automatiquement.
#                  - trouve  -> binaire CPU+GPU (kangaroo_gpu.cu compile avec nvcc)
#                  - absent  -> binaire CPU seul, identique au programme original
#  "make cpu"   : force un binaire CPU seul, meme si nvcc est installe.
#  "make kaggle": raccourci pense pour les notebooks Kaggle (2x T4, sm_75).
#  "make clean" : nettoie les fichiers objets et le binaire.
# =============================================================================
CXX      ?= g++
NVCC     := $(shell command -v nvcc 2>/dev/null)
CXXFLAGS ?= -O3 -march=native -flto -funroll-loops -std=c++17 -pthread
NVCCFLAGS?= -O3 -std=c++17 --extended-lambda -Xcompiler "-O3 -pthread"
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
	$(CXX) $(CXXFLAGS) kangaroo.o kangaroo_gpu.o -o $(BIN) -L$(shell dirname $(NVCC))/../lib64 -lcudart
endif

cpu:
	$(CXX) $(CXXFLAGS) kangaroo.cpp -o $(BIN)

# Kaggle : 2x Nvidia T4 (Turing, sm_75), CUDA deja installe dans l'image standard.
kaggle:
	nvcc -O3 -std=c++17 --extended-lambda -Xcompiler "-O3 -pthread" -gencode arch=compute_75,code=sm_75 -c kangaroo_gpu.cu -o kangaroo_gpu.o
	g++ -O3 -march=native -flto -funroll-loops -std=c++17 -pthread -DHAVE_CUDA -c kangaroo.cpp -o kangaroo.o
	g++ -O3 -march=native -flto -funroll-loops -std=c++17 -pthread kangaroo.o kangaroo_gpu.o -o $(BIN) -lcudart
	@echo "OK : ./kangaroo -gpu all  (utilise les 2 T4 + tous les coeurs CPU)"

clean:
	rm -f $(BIN) kangaroo.o kangaroo_gpu.o
