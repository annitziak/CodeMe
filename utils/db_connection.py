import psycopg2
import logging

from psycopg2 import extras
# from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# load_dotenv()


class DBConnection:
    def __init__(self, db_params: dict):
        self.db_params = db_params

        self.conn = None

    @staticmethod
    def clone(db_conn):
        return DBConnection(db_conn.db_params)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def is_connected(self):
        return self.conn is not None and self.conn.closed == 0

    def get_cursor(self, name: str | None = None):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        if name is not None:
            return self.conn.cursor(name=name)
        return self.conn.cursor()

    def connect(self):
        logger.info(f"Connecting to database with {self.db_params} ...")
        if self.is_connected():
            logger.warning("Already connected to database.")
            return self

        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.cur = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_params}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise e

        return self

    def disconnect(self):
        if self.cur is not None:
            self.cur.close()
            logger.info("Cursor closed.")
        if self.conn is not None:
            self.conn.close()
            logger.info("Connection closed.")

        logger.info("Disconnected from database.")

    def rollback(self):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        self.conn.rollback()
        logger.info("Rollback performed.")

    def commit(self):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        self.conn.commit()
        logger.info("Commit performed.")

    def execute(self, query: str, commit: bool = False):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        try:
            self.cur.execute(query)
            logger.info("Query executed successfully.")
            if commit:
                self.commit()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.rollback()
            raise e

    def execute_values(
        self,
        query: str,
        values: list,
        page_size: int | None = None,
        commit: bool = False,
    ):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        try:
            if page_size is not None:
                extras.execute_values(self.cur, query, values, page_size=page_size)
            else:
                extras.execute_values(self.cur, query, values)
            logger.info("Query executed successfully.")
            if commit:
                self.commit()
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.rollback()
            raise e

    def fetchmany(self, size: int = 1):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        return self.cur.fetchmany(size)

    def fetchall(self):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        return self.cur.fetchall()

    def fetchone(self):
        if not self.is_connected():
            raise Exception("Not connected to database.")

        return self.cur.fetchone()
