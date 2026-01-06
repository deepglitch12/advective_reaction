from torch.utils.data import DataLoader
from datasets import PipeDatasetFromFiles
import time

dataset = PipeDatasetFromFiles(
    dataset_dir="./data/datasets",
    device="cpu"
)

t = time.time()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

for x, y in loader:
    print(x.shape, y.shape)
    print(min(x[:,4]))
    break

print(time.time() - t)
