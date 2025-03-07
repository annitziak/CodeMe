import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_NAME = os.getenv("POSTGRES_DB")
DB_PWD = os.getenv("POSTGRES_PASSWORD")

DB_PARAMS = {"user": DB_USER, "database": DB_NAME, "password": DB_PWD}
