import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

# from kmeans_pytorch import kmeans # training on gpus
from sklearn.cluster import KMeans, SpectralClustering # training on cpu
# assign labels after the Laplacian embedding.
# The cluster_qr method [5] directly extract clusters from eigenvectors in spectral clustering.
# Simple, direct, and efficient multi-way spectral clustering, 2019 Anil Damle, Victor Minden, Lexing Ying

# from .sdp_kmeans import sdp_kmeans, connected_components, spectral_embedding
# https://github.com/simonsfoundation/sdp_kmeans.git
# Mariano Tepper, Anirvan Sengupta, Dmitri Chklovskii, The surprising secret identity of the semidefinite relaxation of K-means: manifold learning, 2017


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []

class GO4ALIGN(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            num_groups: int,
            task_weights: Union[List[float], torch.Tensor] = None,
            robust_step_size: float = 0.0001,
    ):
        super().__init__(n_tasks, device=device)
        self.n_tasks = n_tasks
        self.device = device
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

        self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
        self.robust_step_size = robust_step_size
        self.num_groups = num_groups

    def get_weighted_loss(self, losses, **kwargs):
        adjusted_loss = losses.detach()
        scale = adjusted_loss.sum() / adjusted_loss
        self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())
        weight = scale * self.adv_probs

        weight = weight.unsqueeze(1)
        if self.num_groups >=2:
            # cluster_ids_x, cluster_centers = KMeans(X=weight, num_clusters=self.num_groups, distance='euclidean', device=self.device)
            kmeans = KMeans(n_clusters=self.num_groups, random_state=42)
            cluster_ids_x = kmeans.fit_predict(weight.cpu().numpy()).astype(np.int64)
            cluster_centers = kmeans.cluster_centers_
            mask = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
            cluster_ids = torch.from_numpy(cluster_ids_x).unsqueeze(1).to(self.device)
            cluster_centers = torch.from_numpy(cluster_centers).to(self.device)
            mask.scatter_(1, cluster_ids, 1)
            kmeans_weight = torch.mm(mask,cluster_centers).squeeze(1)

        elif self.num_groups == 1:
            kmeans_weight = torch.ones(self.n_tasks).to(self.device)
            kmeans_weight = kmeans_weight * torch.mean(weight)

        loss = torch.sum(losses * kmeans_weight)
        return loss, dict(weights=torch.cat([kmeans_weight]))

# class GROUP(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#     def get_weighted_loss(self, losses, **kwargs):
#         loss = torch.sum(losses * self.task_weights) * self.n_tasks
#         return loss, dict(weights=torch.cat([self.task_weights]))

# class GROUP_RANDOM(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#     def get_weighted_loss(self, losses, **kwargs):
#         idx = torch.randperm(self.task_weights.shape[0])
#         weights = self.task_weights[idx]
#         loss = torch.sum(losses * weights) * self.n_tasks
#         return loss, dict(weights=torch.cat([weights]))

# class GROUP_sklearn_spectral_clustering_cluster_qr(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             num_groups: int,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         self.device = device
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#         self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
#         self.robust_step_size = robust_step_size
#         self.num_groups = num_groups

#     def get_weighted_loss(self, losses, **kwargs):
#         adjusted_loss = losses.detach()
#         scale = adjusted_loss.sum() / adjusted_loss
#         self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
#         self.adv_probs = self.adv_probs / (self.adv_probs.sum())
#         weight = scale * self.adv_probs

#         weight = weight.unsqueeze(1)

#         if self.num_groups >=2:
#             # pdb.set_trace()
#             results = SpectralClustering(n_clusters=self.num_groups,
#                                          assign_labels='cluster_qr',
#                                          affinity='linear',
#                                          random_state=0).fit(weight.cpu())
#             cluster_ids = results.labels_
#             cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

#             assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
#             cluster_ids = cluster_ids.to(torch.int64)
#             assignment_matrix.scatter_(1, cluster_ids, 1)
#             assignment_matrix = assignment_matrix.to(torch.float64)

#             cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
#             cluster_centers = cluster_centers.unsqueeze(1)

#             group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


#         elif self.num_groups == 1:
#             group_weight = torch.ones(self.n_tasks).to(self.device)
#             group_weight = group_weight * torch.mean(weight)

#         loss = torch.sum(losses * group_weight)
#         return loss, dict(weights=torch.cat([group_weight]))

# class GROUP_sklearn_spectral_clustering_discretize(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             num_groups: int,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         self.device = device
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#         self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
#         self.robust_step_size = robust_step_size
#         self.num_groups = num_groups

#     def get_weighted_loss(self, losses, **kwargs):
#         adjusted_loss = losses.detach()
#         scale = adjusted_loss.sum() / adjusted_loss
#         self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
#         self.adv_probs = self.adv_probs / (self.adv_probs.sum())
#         weight = scale * self.adv_probs

#         weight = weight.unsqueeze(1)

#         if self.num_groups >=2:
#             # pdb.set_trace()
#             results = SpectralClustering(n_clusters=self.num_groups,
#                                          assign_labels='discretize',
#                                          affinity='linear',
#                                          random_state=0).fit(weight.cpu())
#             cluster_ids = results.labels_
#             cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

#             assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
#             cluster_ids = cluster_ids.to(torch.int64)
#             assignment_matrix.scatter_(1, cluster_ids, 1)
#             assignment_matrix = assignment_matrix.to(torch.float64)

#             cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
#             cluster_centers = cluster_centers.unsqueeze(1)

#             group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


#         elif self.num_groups == 1:
#             group_weight = torch.ones(self.n_tasks).to(self.device)
#             group_weight = group_weight * torch.mean(weight)

#         loss = torch.sum(losses * group_weight)
#         return loss, dict(weights=torch.cat([group_weight]))

# class GROUP_sklearn_spectral_clustering_kmeans(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             num_groups: int,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         self.device = device
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#         self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
#         self.robust_step_size = robust_step_size
#         self.num_groups = num_groups

#     def get_weighted_loss(self, losses, **kwargs):
#         adjusted_loss = losses.detach()
#         scale = adjusted_loss.sum() / adjusted_loss
#         self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
#         self.adv_probs = self.adv_probs / (self.adv_probs.sum())
#         weight = scale * self.adv_probs

#         weight = weight.unsqueeze(1)

#         if self.num_groups >=2:
#             # pdb.set_trace()
#             results = SpectralClustering(n_clusters=self.num_groups,
#                                          assign_labels='kmeans',
#                                          affinity='linear',
#                                          random_state=0).fit(weight.cpu())
#             cluster_ids = results.labels_
#             cluster_ids = torch.tensor(cluster_ids).unsqueeze(1).to(self.device)

#             assignment_matrix = torch.zeros(self.n_tasks, self.num_groups).to(self.device)
#             cluster_ids = cluster_ids.to(torch.int64)
#             assignment_matrix.scatter_(1, cluster_ids, 1)
#             assignment_matrix = assignment_matrix.to(torch.float64)

#             cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
#             cluster_centers = cluster_centers.unsqueeze(1)

#             group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1


#         elif self.num_groups == 1:
#             group_weight = torch.ones(self.n_tasks).to(self.device)
#             group_weight = group_weight * torch.mean(weight)

#         loss = torch.sum(losses * group_weight)
#         return loss, dict(weights=torch.cat([group_weight]))

# class GROUP_sdp_clustering(WeightMethod):
#     def __init__(
#             self,
#             n_tasks: int,
#             device: torch.device,
#             num_groups: int,
#             task_weights: Union[List[float], torch.Tensor] = None,
#             robust_step_size: float = 0.0001,
#     ):
#         super().__init__(n_tasks, device=device)
#         self.n_tasks = n_tasks
#         self.device = device
#         if task_weights is None:
#             task_weights = torch.ones((n_tasks,))
#         if not isinstance(task_weights, torch.Tensor):
#             task_weights = torch.tensor(task_weights)
#         assert len(task_weights) == n_tasks
#         self.task_weights = task_weights.to(device)

#         self.adv_probs = torch.ones(n_tasks).to(device) / n_tasks
#         self.robust_step_size = robust_step_size
#         self.num_groups = num_groups

#     def get_weighted_loss(self, losses, **kwargs):
#         adjusted_loss = losses.detach()
#         scale = adjusted_loss.sum() / adjusted_loss
#         self.adv_probs = self.adv_probs * torch.exp(- self.robust_step_size * adjusted_loss)
#         self.adv_probs = self.adv_probs / (self.adv_probs.sum())
#         weight = scale * self.adv_probs

#         weight = weight.unsqueeze(1)

#         if self.num_groups >=2:
#             D, Q = sdp_kmeans(X=weight.cpu().numpy(), n_clusters=self.num_groups, method='cvx')
#             assignment_matrix_init = connected_components(Q)
#             assignment_matrix = torch.tensor(assignment_matrix_init).to(self.device)
#             assignment_matrix = assignment_matrix.transpose(0,1).to(torch.float64)

#             cluster_centers = (assignment_matrix * weight).sum(dim=0) / assignment_matrix.sum(0)
#             cluster_centers = cluster_centers.unsqueeze(1)

#             group_weight = torch.mm(assignment_matrix, cluster_centers).squeeze(1) # 3*2, 2*1

#         elif self.num_groups == 1:
#             group_weight = torch.ones(self.n_tasks).to(self.device)
#             group_weight = group_weight * torch.mean(weight)

#         loss = torch.sum(losses * group_weight)
#         return loss, dict(weights=torch.cat([group_weight]))
