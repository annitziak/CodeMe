import logging
import pprint

from preprocessing import NormalTextBlock, LinkBlock, CodeBlock, Block
from preprocessing.parser import DefaultParserInterface, HTMLParserInterface
from preprocessing.tokenizer import (
    DEFAULT_TOKENIZER_KWARGS,
    DEFAULT_NORMALIZER_OPERATIONS,
)

# from preprocessing.original_tokenizer import Tokenizer
from preprocessing.tokenizer import Tokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def BuildParser(parser_type: str, **kwargs):
    if parser_type == "html":
        return HTMLParserInterface(**kwargs)
    elif parser_type == "raw":
        return DefaultParserInterface(**kwargs)
    else:
        raise ValueError(
            f"Parser type {parser_type} not recognized. Use 'html' or 'raw'"
        )


class Preprocessor:
    def __init__(
        self,
        parser_kwargs={},
        tokenizer_kwargs={
            "text_tokenizer_kwargs": DEFAULT_TOKENIZER_KWARGS,
            "code_tokenizer_kwargs": DEFAULT_TOKENIZER_KWARGS,
            "link_tokenizer_kwargs": DEFAULT_TOKENIZER_KWARGS,
            "text_normalizer_operations": DEFAULT_NORMALIZER_OPERATIONS,
            "code_normalizer_operations": DEFAULT_NORMALIZER_OPERATIONS,
            "link_normalizer_operations": DEFAULT_NORMALIZER_OPERATIONS,
        },
    ):
        self.parser = BuildParser(**parser_kwargs)
        self.tokenizer = Tokenizer(**tokenizer_kwargs)

    def __call__(self, text):
        return self.preprocess(text)

    def preprocess(self, text):
        text_blocks = self.parser.parse(text)
        for text_block in text_blocks:
            tokenized_out = self.tokenizer(text_block)
            text_block.words = tokenized_out.tokenized_text
            text_block.block_length = tokenized_out.original_number_of_words

        return text_blocks


if __name__ == "__main__":
    import argparse

    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-test-data", action="store_true")
    parser.add_argument(
        "--inspect-block", choices=["normal", "link", "code"], default=None
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(parser_kwargs={"parser_type": "html"})
    db_connection = DBConnection(DB_PARAMS)

    inspect_block = Block
    if args.inspect_block is not None:
        if args.inspect_block == "normal":
            inspect_block = NormalTextBlock
        elif args.inspect_block == "link":
            inspect_block = LinkBlock
        else:
            inspect_block = CodeBlock

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
                    text_blocks = preprocessor.preprocess(body)
                    pprint.pp([x for x in text_blocks if isinstance(x, inspect_block)])

                should_continue = input("Continue? [(y)/n]: ")
                if should_continue.lower() == "n":
                    break
    else:
        test_htmls = [
            """<html> printf("The default interface CGPath2D is %s\\n cgPATHcg CGPath cgPATH PATHcg</html>""",
            """<html>\n  <body><p>You àb̰àappleàb̰ should implement <a href="https://api.drupal.org/api/drupal/modules%21node%21node.api.php/function/hook_node_presave/7" rel="nofollow"><code>hook_node_presave</code></a> to set the values you need to change there.</p>\n\n<p>Code sample:</p>\n\n<pre><code>function MODULE_node_presave($node) {\n    if($node-&gt;type === \'MY_NODE_TYPE\') \n        $node-&gt;uid = 1;\n}\n</code></pre>\n</body>\n</html>\n""",
            """
            <html><body>
            <p><strong>Program Description:</strong> </p>                                                                                                                                                                                                                                                                                                                                               <p>I am writing a Java program which initial current directory is /home/user/Desktop. I want to run a bash command "du -s" in "location /home/user/project/" for finding the size of that folder so that I can to use the size of the folder in my project. I cannot post the entire code as it is having some sensitive data. I am just posting the code which is needed.</p>                                                                                                                                                                                                            <p><strong>Here is what I have done:-</strong> </p>

<pre><code>import java.io.*;                                                                                                                                                                  import java.io.BufferedReader;  àpleb̰                                                                                                                                                              import java.io.IOException;                                                                                                                                                                   import java.io.InputStream;                                                                                                                                                                   import java.io.InputStreamReader;                                                                                                                                                             import java.io.File;

            public class Exec_in_cur_dir {
    public static void main(String[] args) {                                                                                                                                                          try {
            StringBuffer output = new StringBuffer();                                                                                                                                                     String Command ="cd /home/user/project";   //Bash Command
                                                                                                                                                                                                          // create a process and execute
            Process p = Runtime.getRuntime().exec(Command);                                                                                                                                               BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                                                                                                                                                                                                          String line = "";

            while ((line = reader.readLine())!= null) {
                output.append(line + "\n");
            }

            System.out.println(output.toString());
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
</code></pre>

<p>So if I execute the program, the output is </p>

<blockquote>
  <p>Cannot run program "cd": error=2.</p>
</blockquote>

<p>But it is working every other commands like </p>

<ul>
<li><code>ls</code></li>
<li><code>df -h</code></li>
<li>etc.</li>
</ul>

<p><strong>My Question:</strong></p>

<p>From the above analysis what I have inferred is my java program cannot be able to change directory. So how can I change the directory path and execute a bash command.</p>
            </html></body>
""",
        ]

        for test_html in test_htmls:
            text_blocks = preprocessor.preprocess(test_html)
            pprint.pp([x for x in text_blocks if isinstance(x, inspect_block)])
            input("Press Enter to continue...")
