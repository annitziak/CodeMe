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
        if idx < len(self.dataset["train"]):
            return self.dataset["train"][idx]["func_string"]
        if idx < len(self.dataset["train"]) + len(self.dataset["test"]):
            inner_idx = idx - len(self.dataset["train"])
            return self.dataset["test"][inner_idx]["func_string"]
        if idx < len(self.dataset["train"]) + len(self.dataset["test"]) + len(
            self.dataset["valid"]
        ):
            inner_idx = idx - len(self.dataset["train"]) - len(self.dataset["test"])
            return self.dataset["valid"][inner_idx]["func_string"]

        raise IndexError("Index out of range")

    def __len__(self):
        return (
            len(self.dataset["train"])
            + len(self.dataset["test"])
            + len(self.dataset["valid"])
        )


if __name__ == "__main__":
    dataset = CodeSearchNetDataset()
    ds = dataset[0]
    print(len(dataset))
