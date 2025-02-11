from preprocessing.sub_tokenizers.main_tokenizer import MainTokenizer


class LinkTokenizer:
    def __init__(self, *args, **kwargs) -> None:
        self.tokenizer = MainTokenizer(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, text, href=None, alt_text=None):
        tokenized_text = self.tokenizer(text)
        href = self.tokenizer(href, is_orderable=False) if href else []
        alt_text = self.tokenizer(alt_text, is_orderable=False) if alt_text else []

        return tokenized_text + href + alt_text
