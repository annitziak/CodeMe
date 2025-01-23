import logging
import pprint

from preprocessing.parser import HTMLParserInterface
from preprocessing.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HTMLPreprocessor:
    def __init__(self, parser_kwargs={}):
        # replace with factory method to build the correct parser
        self.html_parser = HTMLParserInterface(**parser_kwargs)
        self.tokenizer = Tokenizer()

    def preprocess(self, text):
        text_blocks = self.html_parser.parse(text)
        for text_block in text_blocks:
            text_block.words = self.tokenizer.tokenize(text_block)

        return text_blocks


if __name__ == "__main__":
    import argparse

    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-test-data", action="store_true")
    args = parser.parse_args()

    preprocessor = HTMLPreprocessor()
    db_connection = DBConnection(DB_PARAMS)

    if not args.use_test_data:
        with db_connection as conn:
            select_query = "SELECT id, body FROM posts LIMIT 1000"
            conn.execute(select_query, commit=False)
            while True:
                posts = conn.fetchmany(size=1)
                if not posts:
                    logger.info("No more posts to parse.")
                    break

                for post in posts:
                    post_id, body = post
                    print(body)
                    text_blocks = preprocessor.preprocess(body)
                    pprint.pp(text_blocks)

                should_continue = input("Continue? [(y)/n]: ")
                if should_continue.lower() == "n":
                    break
    else:
        test_htmls = [
            """
        <html>
        <p>This is a piece of text with <strong>typing</strong> but also tailing text. And then some extra <strong>text</strong> for laughs.</p>
        </html>
        """,
            """<html>\n  <body><p>You should implement <a href="https://api.drupal.org/api/drupal/modules%21node%21node.api.php/function/hook_node_presave/7" rel="nofollow"><code>hook_node_presave</code></a> to set the values you need to change there.</p>\n\n<p>Code sample:</p>\n\n<pre><code>function MODULE_node_presave($node) {\n    if($node-&gt;type === \'MY_NODE_TYPE\') \n        $node-&gt;uid = 1;\n}\n</code></pre>\n</body>\n</html>\n""",
        ]

        for test_html in test_htmls:
            print(test_html)
            parser.parse(test_html)
            print(parser.get_data())
