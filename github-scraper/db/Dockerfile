FROM postgres:16.2-alpine3.19

VOLUME /var/lib/postgresql/data:./postgres-data

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=postgres

RUN usermod -u 5110 postgres
RUN chown -R postgres:postgres /var/lib/postgresql/data
USER postgres

EXPOSE 5432:5432

CMD ["postgres"]
