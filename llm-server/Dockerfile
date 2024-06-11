FROM python:3.10.14

COPY requirements.txt /server/requirements.txt
RUN pip3 install --no-cache-dir -r /server/requirements.txt

COPY *.py /server/
COPY entrypoint.sh /usr/local/bin/
COPY huggingface-token /server/
COPY out_of_date_test/ /server/out_of_date_test/

RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /server

ENTRYPOINT [ "entrypoint.sh" ]
