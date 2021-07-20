FROM python:3.8 
WORKDIR /vgg19_production
COPY . /vgg19_production
RUN pip install -r requirements.txt
CMD python3 model.py