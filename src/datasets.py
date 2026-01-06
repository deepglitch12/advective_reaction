import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class PipeDatasetFromFiles(Dataset):
    def __init__(self, dataset_dir="../data/datasets", device="cpu"):
        self.device = device
        self.datasets = []
        self.offsets = []

        files = sorted(glob.glob(f"{dataset_dir}/dataset_*.npz"))
        assert len(files) > 0, "No dataset files found."

        offset = 0
        for fname in files:
            data = np.load(fname)

            x = data["x"]
            t = data["t"]
            T = data["T"]
            T_amb = float(data['T_amb'])

            T_in = data["T_in"]
            v = float(data["v"])
            h = float(data["h"])
            D = float(data["D"])
            rho = float(data["rho"])
            cp = float(data["cp"])

            alpha = 4.0 * h / (rho * cp * D)

            Nt = len(t)
            Nx = len(x)
            n_samples = Nt * Nx

            self.datasets.append(
                {
                    "x": x,
                    "t": t,
                    "T": T,
                    "T_in": T_in,
                    "v": v,
                    "alpha": alpha,
                    "T_amb": T_amb,
                    "Nt": Nt,
                    "Nx": Nx,
                    "n": n_samples,
                }
            )

            self.offsets.append(offset)
            offset += n_samples

        self.total_samples = offset

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # find which dataset this idx belongs to
        for d, off in zip(self.datasets, self.offsets):
            if idx < off + d["n"]:
                local_idx = idx - off
                break

        Nt, Nx = d["Nt"], d["Nx"]

        it = local_idx // Nx
        ix = local_idx % Nx

        t    = d["t"][it]
        x    = d["x"][ix]
        T    = d["T"][it, ix]
        T_in = d["T_in"][it]

        X_np = np.array(
            [t, x, T_in , d["v"], d["alpha"], d["T_amb"]],
            dtype=np.float32,
        )

        X = torch.from_numpy(X_np).to(self.device)


        y = torch.tensor([T], dtype=torch.float32, device=self.device)

        return X, y
