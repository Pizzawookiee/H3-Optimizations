FROM nvcr.io/nvidia/pytorch:26.07-py3

RUN python -m pip install --no-cache-dir --upgrade \
    "nvidia-cutlass-dsl[cu13]==4.7.0" \
    "nvidia-cudnn-frontend==1.26.0" \
    "apache-tvm-ffi>=0.1.11" \
    torch-c-dlpack-ext
