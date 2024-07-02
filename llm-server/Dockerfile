FROM python:3.10.14

ARG UID=1000

RUN groupadd -g 9002 llm_group && useradd -m -u $UID -g llm_group llm_user
RUN mkdir /server && chown llm_user:llm_group /server
USER llm_user

WORKDIR /server

COPY --chown=llm_user:llm_group requirements.txt /server/requirements.txt
RUN pip3 install --no-cache-dir -r /server/requirements.txt

ENV PATH="/home/llm_user/.local/bin:${PATH}"

COPY --chown=llm_user:llm_group . /server/
COPY --chown=llm_user:llm_group entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/entrypoint.sh


ENTRYPOINT [ "entrypoint.sh" ]
