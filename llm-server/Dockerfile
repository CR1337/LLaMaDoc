FROM python:3.10.14

COPY requirements.txt /server/requirements.txt
RUN pip3 install --no-cache-dir -r /server/requirements.txt

COPY . /server/
COPY entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /server

ENTRYPOINT [ "entrypoint.sh" ]
