import peewee
from uuid import uuid4


db = peewee.PostgresqlDatabase(
    database="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)


class BaseModel(peewee.Model):
    class Meta:
        database = db

    id = peewee.BinaryUUIDField(primary_key=True, default=uuid4)


class Repository(BaseModel):
    class Meta:
        table_name = "repositories"

    full_name = peewee.CharField()
    size = peewee.IntegerField()
    open_issues = peewee.IntegerField()
    watchers = peewee.IntegerField()
    contributors = peewee.IntegerField()
    forks = peewee.IntegerField()
    contributions = peewee.IntegerField()
    stars = peewee.IntegerField()


class File(BaseModel):
    class Meta:
        table_name = "files"

    repo = peewee.ForeignKeyField(Repository, backref="files")
    path = peewee.CharField()


class Function(BaseModel):
    class Meta:
        table_name = "functions"

    file_ = peewee.ForeignKeyField(File, backref="functions")
    name = peewee.CharField()


class Version(BaseModel):
    class Meta:
        table_name = "versions"

    function = peewee.ForeignKeyField(Function, backref="versions")
    last_version = peewee.ForeignKeyField("self", null=True)
    commit = peewee.CharField()
    date = peewee.DateTimeField()
    code = peewee.TextField()
    docstring = peewee.TextField()
    code_updated = peewee.BooleanField(null=True)
    docstring_updated = peewee.BooleanField(null=True)
    code_similarity = peewee.FloatField(null=True)
    docstring_similarity = peewee.FloatField(null=True)
