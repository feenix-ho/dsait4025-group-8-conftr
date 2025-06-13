# 1) Use NVIDIA’s CUDA 10.1 runtime as the base so you get libcudart.so.10.1 etc.
FROM cld1994/cuda:10.1-cudnn7-devel-ubuntu20.04


# 2) Install system deps and Miniconda
ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# 3) Copy environment spec & your scripts
WORKDIR /workspace
COPY environment.yml .
COPY enhanced_gpu_test.py tf_gpu_check.py conda_test.sh train.sh eval.sh ./

RUN mkdir results

# Create the env (this pulls TF-GPU 2.4.1 + cudatoolkit=10.1)
RUN conda env create -f environment.yml
RUN conda init
RUN conda run -n conformal_training python tf_gpu_check.py

# 6) Default to dropping you into the env
SHELL ["conda", "run", "-n", "conformal_training", "/bin/bash", "-lc"]
CMD ["bash"]
# RUN conda activate conformal_training
# RUN python enhanced_gpu_test.py

# # Activate by default
# CMD ["conda", "run", "-n", conformal_training, "/bin/bash", "-lc"]
