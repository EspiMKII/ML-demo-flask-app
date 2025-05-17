# base image
FROM python:3.13.1-slim
WORKDIR /app

# dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# source code; might need to copy more later so don't imagerize yet!
COPY MLApp ./MLApp/
COPY app.py ./

# expose flask port
EXPOSE 5000

# command run on container start
CMD ["flask", "--app", "MLApp",  "run",  "--host", "0.0.0.0", "--port", "5000"]