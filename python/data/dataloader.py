from torch.utils.data import DataLoader


def build_loader(dataset, batch_size: int = 100, sampler: str = "base"):
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader
