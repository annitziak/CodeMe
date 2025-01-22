import logging
from lxml import etree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HTMLParserInterface:
    def __init__(self):
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)

    def feed(self, data: str):
        self.parser.feed(data)

    def add_data(self, data: str):
        self.root = etree.fromstring(data, self.parser)

    def get_data(self) -> str:
        return self.root.tostring(self.root, pretty_print=True)


if __name__ == "__main__":
    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = HTMLParserInterface()
    db_connection = DBConnection(DB_PARAMS)

    select_query = "SELECT id, body FROM posts LIMIT 5"
    with db_connection as conn:
        conn.execute(select_query)
        posts = conn.fetchall()

    for post in posts:
        post_id, body = post
        parser.feed(body)
        print(f"Post ID: {post_id}")
        print(f"Body: {parser.get_data()}")
        print("\n")
