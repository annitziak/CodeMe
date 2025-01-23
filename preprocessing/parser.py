import logging

from preprocessing import CodeBlock, LinkBlock, NormalTextBlock, TextSize
from lxml import etree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HTMLParserInterface:
    def __init__(self):
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)
        self.text_blocks = []

    def feed(self, data: str):
        self.parser.feed(data)

    def parse(self, data: str):
        self.root = etree.fromstring(data, self.parser)
        self.text_blocks = sorted(
            self.process_element(self.root)[0], key=lambda x: x.block_id
        )

        return self.text_blocks

    def process_element(self, element: etree.Element, parent_element=None, id=0):
        if element is None:
            return

        element_result, tailing_text = self._handle_element(
            element, parent_element=parent_element, id=id
        )
        element_results = []

        child_id = id
        for child in element:
            child_id += 1

            child_element_result, child_tailing_text = self.process_element(
                child, parent_element=element, id=child_id
            )
            if child_element_result is not None:
                element_results.extend(child_element_result)

            child_id = (
                element_results[-1].block_id if len(element_results) > 0 else child_id
            )

            if child_tailing_text is not None and element_result is not None:
                child_id += 1
                secondary_text_block, _ = self._handle_tag(
                    tag=element.tag, text=child_tailing_text, id=child_id
                )
                if secondary_text_block is not None:
                    element_results.append(secondary_text_block)

        if element_result is not None:
            element_results.insert(0, element_result)

        return element_results, tailing_text

    def get_data(self) -> str:
        return etree.tostring(self.root, pretty_print=True)

    def _handle_element(self, element: etree.Element, parent_element=None, **kwargs):
        return self._handle_tag(
            tag=element.tag,
            text=element.text,
            tail=element.tail,
            parent_tag=parent_element.tag if parent_element is not None else "",
            attrib=element.attrib,
            **kwargs,
        )

    def _handle_tag(
        self,
        tag: str = "p",
        text: str = "",
        tail: str = "",
        parent_tag: str = "",
        id: int = -1,
        **kwargs,
    ):
        if tag == "code":
            if parent_tag == "pre":
                text_block = CodeBlock(text=text, block_id=id, in_line=False)
            else:
                text_block = CodeBlock(text=text, block_id=id, in_line=True)
        elif tag in ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                text_size=TextSize[tag.upper()],
            )
        elif tag == "a":
            text_block = LinkBlock(
                text=text,
                block_id=id,
                href=kwargs.get("attrib", {}).get("href", ""),
                alt_text=kwargs.get("attrib", {}).get("alt", ""),
                text_size=TextSize[parent_tag.upper()],
            )
        elif tag in ["strong", "b"]:
            text_block = NormalTextBlock(text=text, block_id=id, is_bold=True)
        elif tag in ["em", "i"]:
            text_block = NormalTextBlock(text=text, block_id=id, is_italic=True)
        elif tag == "u":
            text_block = NormalTextBlock(text=text, block_id=id, is_underline=True)
        else:
            return None, None

        return text_block, tail

    def _get_parent(self, element: etree.Element):
        parent = element.getparent()
        if parent is None:
            return None

        return parent


if __name__ == "__main__":
    import argparse

    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-test-data", action="store_true")
    args = parser.parse_args()

    parser = HTMLParserInterface()
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
                    parser.parse(body)
                    print(f"Post ID: {post_id}")
                    print(f"Body: {parser.get_data()}")
                    print("\n")

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
            parser.parse(test_html)
            print(parser.get_data())
            print(test_html)
