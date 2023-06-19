FROM python:3.9.7
WORKDIR /code/
COPY requirements.txt /code/requirements.txt

RUN python3 -m pip install -r requirements.txt

COPY . /code/

EXPOSE 8000

CMD ["python3", "app.py"]

# RUN python3 -m pip install pytest

# ssh
# ENV SSH_PASSWD "root:Docker!"
# RUN apt-get update \
#         && apt-get install -y --no-install-recommends dialog \
#         && apt-get update \
#         && apt-get install -y --no-install-recommends openssh-server \
#         && echo "$SSH_PASSWD" | chpasswd 

# COPY sshd_config /etc/ssh/
# COPY init.sh /usr/local/bin/
	
# RUN chmod u+x /usr/local/bin/init.sh
# EXPOSE 8000 2222

# ENTRYPOINT ["init.sh"]
# ENTRYPOINT ["python", "app.py"]
