from argparse import ArgumentParser
import os

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import trange
import wandb
import time

from experiments.quantum_chemistry.models import Net
from experiments.quantum_chemistry.utils import (
    Complete,
    MyTransform,
    delta_fn,
    multiply_indx,
    delta_fn_arr,
)
from experiments.quantum_chemistry.utils import target_idx as targets
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

set_logger()


@torch.no_grad()
def evaluate(model, loader, std, scale_target):
    model.eval()
    data_size = 0.0
    task_losses = 0.0
    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        if scale_target:
            task_losses += F.l1_loss(
                out * std.to(device), data.y * std.to(device), reduction="none"
            ).sum(
                0
            )  # MAE
        else:
            task_losses += F.l1_loss(out, data.y, reduction="none").sum(0)  # MAE
        data_size += len(data.y)

    model.train()

    avg_task_losses = task_losses / data_size

    # Report meV instead of eV.
    avg_task_losses = avg_task_losses.detach().cpu().numpy()
    avg_task_losses[multiply_indx] *= 1000

    delta_m = delta_fn(avg_task_losses)
    scores = delta_fn_arr(avg_task_losses)

    return dict(
        avg_loss=avg_task_losses.mean(),
        avg_task_losses=avg_task_losses,
        delta_m=delta_m,
        scores=scores,
    )


def main(
    data_path: str,
    batch_size: int,
    device: torch.device,
    method: str,
    weight_method_params: dict,
    lr: float,
    lr_patience: int,
    method_params_lr: float,
    n_epochs: int,
    targets: list = None,
    scale_target: bool = True,
    main_task: int = None,
    ckpt_path: str = None,
):
    dim = 64
    model = Net(n_tasks=len(targets), num_features=11, dim=dim).to(device)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        logging.info(f"Loading model weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        logging.info("Training from scratch.")
    transform = T.Compose([MyTransform(targets), Complete(), T.Distance(norm=False)])
    dataset = QM9(data_path, transform=transform).shuffle()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]

    std = None
    if scale_target:
        mean = train_dataset.data.y[:, targets].mean(dim=0, keepdim=True)
        std = train_dataset.data.y[:, targets].std(dim=0, keepdim=True)

        dataset.data.y[:, targets] = (dataset.data.y[:, targets] - mean) / std

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    weight_method = WeightMethods(
        method,
        n_tasks=len(targets),
        device=device,
        **weight_method_params[method],
    )
    print(f"number of tasks: {len(targets)}")

    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=method_params_lr),
        ],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=lr_patience, min_lr=0.00001
    )

    epoch_iterator = trange(n_epochs)

    best_val = np.inf
    best_test = np.inf
    best_test_delta = np.inf
    best_val_delta = np.inf
    best_test_results = None

    avg_cost = np.zeros([n_epochs, 13*2], dtype=np.float32)
    deltas = np.zeros([n_epochs], dtype=np.float32)
    # save model
    save_model_dir = os.path.join('./save', args.wandb_logger_name)
    os.makedirs(save_model_dir, exist_ok=True)

    # some extra statistics we save during training
    loss_list = []
    scores = np.array([0.0] * len(targets), dtype=np.float32)
    tmp_scores = np.array([0.0] * len(targets), dtype=np.float32)
    for epoch in epoch_iterator:
        lr = scheduler.optimizer.param_groups[0]["lr"]
        for j, data in enumerate(train_loader):
            model.train()

            data = data.to(device)
            optimizer.zero_grad()

            out, features = model(data, return_representation=True)

            losses = F.mse_loss(out, data.y, reduction="none").mean(0)

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
                scores=scores,
            )

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    out_ = model(data, return_representation=False)
                    new_losses = F.mse_loss(out_, data.y, reduction="none").mean(0)
                    weight_method.method.update(new_losses.detach())

        val_loss_dict = evaluate(model, val_loader, std=std, scale_target=scale_target)
        test_loss_dict = evaluate(
            model, test_loader, std=std, scale_target=scale_target
        )
        val_loss = val_loss_dict["avg_loss"]
        val_delta = val_loss_dict["delta_m"]

        test_loss = test_loss_dict["avg_loss"]
        test_delta = test_loss_dict["delta_m"]

        scores = val_loss_dict["scores"]

        if method == "stl":
            best_val_criteria = val_loss_dict["avg_task_losses"][main_task] <= best_val
        else:
            best_val_criteria = val_delta <= best_val_delta

        if best_val_criteria:
            best_val = val_loss
            best_test = test_loss
            best_test_results = test_loss_dict
            best_val_delta = val_delta
            best_test_delta = test_delta

        avg_cost[epoch,0] = val_loss
        avg_cost[epoch,1] = val_delta
        avg_cost[epoch,2:2+11] = val_loss_dict["avg_task_losses"]
        avg_cost[epoch,13] = test_loss
        avg_cost[epoch,14:14+11] = test_loss_dict["avg_task_losses"]
        deltas[epoch] = test_delta

        # for logger
        epoch_iterator.set_description(
            f"epoch {epoch} | lr={lr} | train loss {losses.mean().item():.3f} | val loss: {val_loss:.3f}\n"
            f"test loss: {test_loss:.3f} | best test loss {best_test:.3f} | best_test_delta {best_test_delta:.3f}\n"
        )
        for i in range(len(targets)):
           print(
                f"Epoch: {epoch}, Test Loss: {test_loss_dict['avg_task_losses'][i]:.3f}\n" 
            )

        if wandb.run is not None:
            wandb.log({"Learning Rate": lr}, step=epoch)
            wandb.log({"Train Loss": losses.mean().item()}, step=epoch)
            wandb.log({"Val Loss": val_loss}, step=epoch)
            wandb.log({"Val Delta": val_delta}, step=epoch)
            wandb.log({"Test Loss": test_loss}, step=epoch)
            wandb.log({"Test Delta": test_delta}, step=epoch)
            wandb.log({"Best Test Loss": best_test}, step=epoch)
            wandb.log({"Best Test Delta": best_test_delta}, step=epoch)

        scheduler.step(
            val_loss_dict["avg_task_losses"][main_task]
            if method == "stl"
            else val_delta
        )

        # save state_dict
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_model_dir, f"{args.method}_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch}.")
        

        name = f"{args.wandb_logger_name}"

        torch.save({
            "avg_cost": avg_cost,
            "losses": loss_list,
            "delta_m": deltas,
        }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("QM9", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-3,
        n_epochs=300,
        batch_size=60,
        method="nashmtl",
    )
    parser.add_argument("--scale-y", default=False, type=str2bool)
    parser.add_argument("--lr_patience", type=int, default=5, help="Patience for LR scheduler.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint file path to load model weights from.")
    parser.add_argument("--wandb_logger_name", type=str, default="QM9", help="Name of Weights & Biases Logger.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(name=args.wandb_logger_name, project=args.wandb_project, entity=args.wandb_entity, config=args)

    weight_method_params = extract_weight_method_parameters_from_args(args)

    device = get_device(gpus=args.gpu)
    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        method=args.method,
        weight_method_params=weight_method_params,
        lr=args.lr,
        lr_patience=args.lr_patience,
        method_params_lr=args.method_params_lr,
        n_epochs=args.n_epochs,
        targets=targets,
        scale_target=args.scale_y,
        main_task=args.main_task,
        ckpt_path=args.ckpt,
    )

    if wandb.run is not None:
        wandb.finish()
