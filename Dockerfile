FROM python:3.7.3-stretch

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY app.py /app/
COPY requirements.txt /app/

RUN mkdir /app/templates
COPY templates/upload.html /app/templates/

# Install packages from requirements.txt
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8848
EXPOSE 8848

# Run app.py at container launch
CMD ["python", "app.py"]