FROM nvcr.io/nvidia/pytorch:21.07-py3
RUN ["apt-get", "update"]
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --upgrade
WORKDIR /workspace/efm
ENV PYTHONPATH "${PYTHONPATH}:./"
RUN pip uninstall -y opencv-python
RUN pip uninstall -y opencv-contrib-python
RUN pip uninstall -y opencv-python-headless
RUN pip install opencv-python==4.5.5.64
RUN pip install opencv-contrib-python==4.5.5.64
RUN pip install opencv-python-headless==4.5.5.64
RUN pip install scikit-image -U