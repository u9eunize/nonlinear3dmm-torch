FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt install -y python-pip python-dev build-essential ssh openssh-server vim

## SSH
RUN echo "root:tmax1243!@" | chpasswd
COPY sshd_config /etc/ssh/sshd_config
EXPOSE 40000

RUN useradd -ms /bin/bash tmax
RUN echo "tmax:tmax1234" | chpasswd
RUN service ssh start
RUN service ssh restart

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

#COPY app/build /tmp/build/
#WORKDIR /tmp/build/
#RUN python setup.py install


## APP
COPY app /app
WORKDIR /app

# RUN mkdir /data