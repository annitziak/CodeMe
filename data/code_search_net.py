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
        print(self.dataset["train"][idx])
        return self.dataset["train"][idx]["whole_func_string"]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = CodeSearchNetDataset()
    ds = dataset[0]
    print(len(dataset))
