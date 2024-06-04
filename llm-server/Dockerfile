FROM huggingface/transformers-pytorch-gpu:4.41.0

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY *.py /server/
COPY token_frequencies.pkl /server/token_frequencies.pkl

ENTRYPOINT [ "entrypoint.sh" ]
