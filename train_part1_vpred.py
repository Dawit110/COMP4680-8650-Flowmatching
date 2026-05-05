from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataloader import ToyDiffusionDataset, get_dataloader
from src.model import MLPVectorField


def sample_with_euler(model, n_samples: int, dim: int, steps: int, device: str):
    """
    Generate samples by integrating backwards from noise at t=1 to data at t=0.

    Forward flow:
        data x at t=0 -> noise eps at t=1

    Generation:
        start from noise and move backwards.
    """
    model.eval()

    z = torch.randn(n_samples, dim, device=device)
    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):
            t_value = 1.0 - i * dt
            t = torch.full((n_samples,), t_value, device=device)

            v = model(z, t)

            # Backwards Euler step: move from t toward 0.
            z = z - dt * v

    return z


def plot_samples(dataset, generated, output_path: Path):
    real = dataset.data.numpy()
    generated = generated.cpu().numpy()

    real_2d = dataset.to_2d(real)
    generated_2d = dataset.to_2d(generated)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(real_2d[:, 0], real_2d[:, 1], s=2, alpha=0.5)
    plt.title("Real data")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.scatter(generated_2d[:, 0], generated_2d[:, 1], s=2, alpha=0.5)
    plt.title("Generated samples")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure to {output_path}")


def train(args):
    device = "cpu"

    Path("checkpoints").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    dataset = ToyDiffusionDataset(name=args.dataset, dim=args.dim)
    dataloader = get_dataloader(
        name=args.dataset,
        dim=args.dim,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = MLPVectorField(dim=args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_iter = iter(dataloader)

    for step in range(1, args.steps + 1):
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        x = x.float().to(device)

        batch_size = x.shape[0]

        # Sample noise and random time.
        eps = torch.randn_like(x)
        t = torch.rand(batch_size, device=device)

        # Linear interpolation from data x to noise eps.
        z_t = (1.0 - t[:, None]) * x + t[:, None] * eps

        # v-prediction target.
        target_v = eps - x

        pred_v = model(z_t, t)

        loss = F.mse_loss(pred_v, target_v)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:05d} | loss {loss.item():.6f}")

    ckpt_path = Path("checkpoints") / f"part1_vpred_{args.dataset}_{args.dim}d.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": args.dataset,
            "dim": args.dim,
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    generated = sample_with_euler(
        model=model,
        n_samples=args.n_samples,
        dim=args.dim,
        steps=args.euler_steps,
        device=device,
    )

    fig_path = Path("figures") / f"part1_vpred_{args.dataset}_{args.dim}d_samples.png"
    plot_samples(dataset, generated, fig_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="swiss_roll")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--euler-steps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=500)

    args = parser.parse_args()

    if args.dim != 2:
        raise ValueError("Part 1 baseline should first be run with --dim 2")

    train(args)


if __name__ == "__main__":
    main()