from datasets import load_dataset


class CodeDataset:
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CodeSearchNetDataset(CodeDataset):
    def __init__(self):
        self.dataset = load_dataset("code_search_net")

    def get_dataset(self):
        return CodeDataset(self.dataset["train"])

    def __getitem__(self, idx):
        start_idx = idx if isinstance(idx, int) else idx.start
        end_idx = idx if isinstance(idx, int) else idx.stop

        ret_items = []
        for i in range(start_idx, min(end_idx, len(self))):
            if i < len(self.dataset["train"]):
                item = self.dataset["train"][i]
                text = " ".join(
                    [item["func_code_string"], item["func_documentation_string"]]
                )
                ret_items.append(text)
            else:
                item = self.dataset["test"][i - len(self.dataset["train"])]
                text = " ".join(
                    [item["func_code_string"], item["func_documentation_string"]]
                )
                ret_items.append(text)

        return ret_items

    def __len__(self):
        return len(self.dataset["train"]) + len(self.dataset["test"])


if __name__ == "__main__":
    dataset = CodeSearchNetDataset()
    ds = dataset[0]
    print(len(dataset))
