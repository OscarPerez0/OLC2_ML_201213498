FROM python:3.9

EXPOSE 8501

WORKDIR /app

COPY  requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install streamlit
RUN pip install pandas
RUN pip install xlrd
RUN pip install matplotlib
RUN pip install numpy
RUN pip install -U scikit-learn
RUN pip install scikit-metrics

RUN git clone https://github.com/OscarPerez0/OLC2_ML_201213498.git

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "./OLC2_ML_201213498/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]