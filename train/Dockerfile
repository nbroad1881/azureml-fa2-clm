FROM nvcr.io/nvidia/pytorch:23.08-py3

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip uninstall -y flash-attn
RUN MAX_JOBS=4 pip install flash-attn==2.5.2 --no-build-isolation
