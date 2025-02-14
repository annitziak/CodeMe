import logging

from preprocessing import CodeBlock, LinkBlock, NormalTextBlock, TextSize
from lxml import etree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DefaultParserInterface:
    def __init__(self):
        self.text_blocks = []

    def parse(self, data: str):
        text_block = NormalTextBlock(text=data, block_id=0, text_size=TextSize.P)
        self.text_blocks = [text_block]

        return self.text_blocks


class HTMLParserInterface:
    def __init__(self):
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)
        self.text_blocks = []

    def __getstate__(self) -> object:
        data = self.__dict__.copy()
        del data["parser"]
        del data["root"]

        return data

    def __setstate__(self, state: object):
        self.__dict__.update(state)
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)

    def feed(self, data: str):
        self.parser.feed(data)

    def parse(self, data: str):
        self.root = etree.fromstring(data, self.parser)
        self.text_blocks, _ = self.process_element(self.root)
        self.text_blocks = sorted(self.text_blocks, key=lambda x: x.block_id)

        return self.text_blocks

    def process_element(
        self, element: etree.Element, parent_element=None, parent_block=None, id=0
    ):
        if element is None:
            return [], None

        element_result, tailing_text = self._handle_element(
            element, parent_element=parent_element, parent_block=parent_block, id=id
        )

        element_results = []

        child_id = id
        for child in element:
            child_id += 1

            child_element_result, child_tailing_text = self.process_element(
                child, parent_element=element, parent_block=element_result, id=child_id
            )
            if child_element_result is not None:
                element_results.extend(child_element_result)

            child_id = (
                element_results[-1].block_id if len(element_results) > 0 else child_id
            )

            if child_tailing_text is not None and element_result is not None:
                child_id += 1
                secondary_text_block, _ = self._handle_tag(
                    tag=element.tag,
                    text=child_tailing_text,
                    id=child_id,
                    parent_tag=(
                        parent_element.tag if parent_element is not None else ""
                    ),
                    parent_block=parent_block,
                )
                if secondary_text_block is not None:
                    element_results.append(secondary_text_block)

        if element_result is not None:
            element_results.insert(0, element_result)

        return element_results, tailing_text

    def get_data(self) -> str:
        return etree.tostring(self.root, pretty_print=True)

    def _handle_element(
        self, element: etree.Element, parent_element=None, parent_block=None, **kwargs
    ):
        return self._handle_tag(
            tag=element.tag,
            text=element.text,
            tail=element.tail,
            parent_tag=parent_element.tag if parent_element is not None else "",
            parent_block=parent_block,
            attrib=element.attrib,
            **kwargs,
        )

    def _handle_tag(
        self,
        tag: str = "p",
        text: str = "",
        tail: str = "",
        parent_tag: str = "",
        parent_block=None,
        id: int = -1,
        **kwargs,
    ):
        if tag == "code":
            if parent_tag == "pre":
                text_block = CodeBlock(
                    text=text,
                    block_id=id,
                    in_line=False,
                ).update(parent_block)
            else:
                text_block = CodeBlock(
                    text=text,
                    block_id=id,
                    in_line=True,
                ).update(parent_block)
        elif tag in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                text_size=self._get_text_size(tag),
            ).update(parent_block)
        elif tag == "a":
            text_block = LinkBlock(
                text=text,
                block_id=id,
                href=kwargs.get("attrib", {}).get("href", ""),
                alt_text=kwargs.get("attrib", {}).get("alt", ""),
                text_size=self._get_text_size(parent_tag),
            ).update(parent_block)
        elif tag in ["strong", "b"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_bold=True,
            ).update(parent_block)
        elif tag in ["em", "i"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_italic=True,
            ).update(parent_block)
        elif tag == "u":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_underline=True,
            ).update(parent_block)
        elif tag == "sup":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_superscript=True,
            ).update(parent_block)
        elif tag in ["s", "del"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_strike_through=True,
            ).update(parent_block)
        elif tag in ["li", "ul", "ol"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_list=True,
            ).update(parent_block)
        elif tag == "blockquote":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_blockquote=True,
            ).update(parent_block)
        elif tag in ["ul", "ol", "li"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_list=True,
            ).update(parent_block)
        elif text in ["dd", "dt", "dl"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_desciption_list=True,
            ).update(parent_block)
        else:
            return None, None

        return text_block, tail

    def _get_parent(self, element: etree.Element):
        parent = element.getparent()
        if parent is None:
            return None

        return parent

    def _get_text_size(self, tag: str):
        upper_tag = tag.upper()
        if upper_tag in TextSize.__members__:
            return TextSize[upper_tag]

        return TextSize.UNK


if __name__ == "__main__":
    import argparse
    import pprint

    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-test-data", action="store_true")
    args = parser.parse_args()

    parser = HTMLParserInterface()
    db_connection = DBConnection(DB_PARAMS)

    with open(".cache/test_documents.txt", "w") as f:
        f.write("id\ttitle\tbody\n")

    if not args.use_test_data:
        with db_connection as conn:
            select_query = "SELECT id, title, body FROM posts WHERE ID=419044 OR ID=419085 LIMIT 1000"
            conn.execute(select_query, commit=False)
            while True:
                posts = conn.fetchmany(size=1)
                if not posts:
                    logger.info("No more posts to parse.")
                    break

                for post in posts:
                    post_id, title, body = post
                    text_blocks = parser.parse(body)
                    print(f"Post ID: {post_id}")
                    print(f"Body:\n {text_blocks}")

                    title = title + " " if title is not None else ""
                    print(title + " ".join([x.text for x in text_blocks]))
                    print("\n")

                    with open(".cache/test_documents.txt", "a") as f:
                        f.write(
                            f"{post_id}\t{title}\t{' '.join([x.text for x in text_blocks])}\n"
                        )

                should_continue = input("Continue? [(y)/n]: ")
                if should_continue.lower() == "n":
                    break
    else:
        test_htmls = [
            """
        <html>
        <h2>Test the handling fo <strong>strong or <i>bold and italic</i></strong> text</h2>
        <p>This is a piece of text with <strong>typing</strong> but also tailing text. And then some extra <strong>text</strong> for laughs.</p>
        </html>
        """,
            """<html>\n  <body><p>You should implement <a href="https://api.drupal.org/api/drupal/modules%21node%21node.api.php/function/hook_node_presave/7" rel="nofollow"><code>hook_node_presave</code></a> to set the values you need to change there.</p>\n\n<p>Code sample:</p>\n\n<pre><code>function MODULE_node_presave($node) {\n    if($node-&gt;type === \'MY_NODE_TYPE\') \n        $node-&gt;uid = 1;\n}\n</code></pre>\n</body>\n</html>\n""",
        ]

        for test_html in test_htmls:
            text_blocks = parser.parse(test_html)
            pprint.pp(text_blocks)
            print(test_html)
