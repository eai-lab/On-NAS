""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""


import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
import scipy.special
import copy

from models import ops_naive
from utils import genotypes as gt

def SoftMax(logits, params, dim=-1):

    # temperature annealing
    if params["temp_anneal_mode"] == "linear":
        # linear temperature annealing (Note: temperature -> zero results in softmax -> argmax)
        temperature = params["t_max"] - params["curr_step"] * (
            params["t_max"] - params["t_min"]
        ) / (params["max_steps"] - 1)
        assert temperature > 0
    else:
        temperature = 1.0
    return F.softmax(logits / temperature, dim=dim)


def ReLUSoftMax(logits, params, dim=-1):
    lamb = params["curr_step"] / (params["max_steps"] - 1)
    return (1.0 - lamb) * F.softmax(logits, dim=dim) + lamb * (
        F.relu(logits) / torch.sum(F.relu(logits), dim=dim, keepdim=True)
    )


def GumbelSoftMax(logits, params, dim=-1):
    # NOTE: dim argument does not exist for gumbel_softmax in pytorch 1.0.1

    # temperature annealing
    if params["temp_anneal_mode"] == "linear":
        # linear temperature annealing (Note: temperature -> zero results in softmax -> argmax)
        temperature = params["t_max"] - params["curr_step"] * (
            params["t_max"] - params["t_min"]
        ) / (params["max_steps"] - 1)
        assert temperature > 0
    else:
        temperature = 1.0
    return F.gumbel_softmax(logits, temperature)


class SearchCNNController_puredarts(nn.Module):
    """ SearchCNN controller supporting multi-gpu """

    def __init__(
        self,
        C_in,
        C,
        n_classes,
        n_layers,
        n_nodes=4,
        reduction_layers=[],
        stem_multiplier=3,
        device_ids=None,
        normalizer=dict(),
        PRIMITIVES=None,
        feature_scale_rate=2,
        use_hierarchical_alphas=False,  # deprecated
        use_pairwise_input_alphas=False,
        alpha_prune_threshold=0.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = nn.CrossEntropyLoss()
        self.use_pairwise_input_alphas = use_pairwise_input_alphas
        self.use_hierarchical_alphas = use_hierarchical_alphas
        self.alpha_prune_threshold = alpha_prune_threshold

        if "name" not in normalizer.keys():
            normalizer["func"] = SoftMax
            normalizer["params"] = dict()
            normalizer["params"]["temp_anneal_mode"] = None
        elif normalizer["name"] == "softmax":
            normalizer["func"] = SoftMax
        elif normalizer["name"] == "relusoftmax":
            normalizer["func"] = ReLUSoftMax
        elif normalizer["name"] == "gumbel_softmax":
            normalizer["func"] = GumbelSoftMax
        else:
            raise RuntimeError(f"Unknown normalizer {normalizer['name']}")
        self.normalizer = normalizer

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        if PRIMITIVES is None:
            PRIMITIVES = gt.PRIMITIVES

        self.primitives = PRIMITIVES
        n_ops = len(PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            # create alpha parameters over parallel operations
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)))

        assert not (
            use_hierarchical_alphas and use_pairwise_input_alphas
        ), "Hierarchical and pairwise alphas exclude each other."

        self.alpha_pw_normal = None
        self.alpha_pw_reduce = None
        self.alpha_in_normal = None
        self.alpha_in_reduce = None
        if use_hierarchical_alphas:  # deprecated
            # create alpha parameters the different input nodes for a cell, i.e. for each node in a
            # cell an additional distribution over the input nodes is introduced
            print("Using hierarchical alphas.")

            self.alpha_in_normal = nn.ParameterList()
            self.alpha_in_reduce = nn.ParameterList()

            for i in range(n_nodes):
                self.alpha_in_normal.append(nn.Parameter(1e-3 * torch.randn(i + 2)))
                self.alpha_in_reduce.append(nn.Parameter(1e-3 * torch.randn(i + 2)))

  
        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if "alpha" in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(
            C_in,
            C,
            n_classes,
            n_layers,
            n_nodes,
            reduction_layers,
            stem_multiplier,
            PRIMITIVES=self.primitives,
            feature_scale_rate=feature_scale_rate,
        )

    def apply_normalizer(self, alpha):
        return self.normalizer["func"](alpha, self.normalizer["params"])

    def _get_normalized_alphas(self):
        weights_normal = [self.apply_normalizer(alpha) for alpha in self.alpha_normal]
        weights_reduce = [self.apply_normalizer(alpha) for alpha in self.alpha_reduce]

        weights_pw_normal = None
        weights_pw_reduce = None
        weights_in_normal = None
        weights_in_reduce = None
        if self.alpha_in_normal is not None:
            weights_in_normal = [
                self.apply_normalizer(alpha) for alpha in self.alpha_in_normal
            ]
            weights_in_reduce = [
                self.apply_normalizer(alpha) for alpha in self.alpha_in_reduce
            ]
        elif self.alpha_pw_normal is not None:
            weights_pw_normal = [
                self.apply_normalizer(alpha) for alpha in self.alpha_pw_normal
            ]
            weights_pw_reduce = [
                self.apply_normalizer(alpha) for alpha in self.alpha_pw_reduce
            ]

        return (
            weights_normal,
            weights_reduce,
            weights_in_normal,
            weights_in_reduce,
            weights_pw_normal,
            weights_pw_reduce,
        )

    def prune_alphas(self, prune_threshold=0.0, val=-10e8):
        """Set the alphas with probability below prune_threshold to a large negative value

        Note:
            The prune_threshold applies to the alpha probabilities (after the softmax is
            applied) while `val` corresponds to the logit values (thus a large negative value
            corresponds to a low probability).
        """

        # reset temperature for prunning
        model_has_normalizer = hasattr(self, "normalizer")
        if model_has_normalizer:
            curr_step_backup = self.normalizer["params"]["curr_step"]
            self.normalizer["params"]["curr_step"] = (
                self.normalizer["params"]["max_steps"] - 1
            )

        weights_normal = [self.apply_normalizer(alpha) for alpha in self.alpha_normal]
        weights_reduce = [self.apply_normalizer(alpha) for alpha in self.alpha_reduce]
        for idx in range(len(weights_normal)):
            # need to modify data because alphas are leaf variables
            self.alpha_normal[idx].data[weights_normal[idx] < prune_threshold] = val
            self.alpha_reduce[idx].data[weights_reduce[idx] < prune_threshold] = val

        # set curr_step back to original value
        self.normalizer["params"]["curr_step"] = curr_step_backup

    def get_sparse_alphas_pw(self, alpha_prune_threshold=0.0):

        """
        Convert alphas to zero-one-vectors under consideration of pairwise alphas


        :param alpha_prune_threshold: threshold for pruning

        :return: binary tensors with shape like alpha_normal and alpha_reduce, indicating whether an op is included in the
        sparsified one shot model
        """

        assert (
            self.alpha_pw_normal is not None
        ), "Error: function only availaible for pw models"

        weights_normal = [
            self.apply_normalizer(alpha) for alpha in self.alpha_normal
        ]  # get normalized weights
        weights_reduce = [self.apply_normalizer(alpha) for alpha in self.alpha_reduce]

        weights_pw_normal = [
            self.apply_normalizer(alpha) for alpha in self.alpha_pw_normal
        ]

        weights_pw_reduce = [
            self.apply_normalizer(alpha) for alpha in self.alpha_pw_reduce
        ]

        weights_normal_sparse = list()

        # get all the pairs of inputs
        for node_idx, node_weights in enumerate(weights_normal):
            input_pairs = list()

            # get pairs of inputs correspeonding to indices in alpha_pw
            for input_1 in range(len(node_weights)):
                for input_2 in range(input_1 + 1, len(node_weights)):
                    input_pairs.append([input_1, input_2])

            assert len(input_pairs) == len(
                weights_pw_normal[node_idx]
            ), "error: pairwise alpha length does not match pairwise terms length"

            keep_inputs = list()  # list of input nodes that are kept

            for input_pair_idx in range(len(input_pairs)):
                if (
                    weights_pw_normal[node_idx][input_pair_idx] >= alpha_prune_threshold
                ):  # if pw weight larger than threshold keep input
                    keep_inputs.extend(input_pairs[input_pair_idx])

            weights_normal_sparse.append(
                torch.stack(
                    [
                        (weight >= alpha_prune_threshold).type(torch.float)
                        if weight_idx in keep_inputs
                        else torch.zeros_like(weight)
                        for weight_idx, weight in enumerate(node_weights)
                    ]
                )
            )

        ### same for reduction

        weights_reduce_sparse = list()

        for node_idx, node_weights in enumerate(weights_reduce):
            input_pairs = list()

            # get pairs of inputs correspeonding to indices in alpha_pw
            for input_1 in range(len(node_weights)):
                for input_2 in range(input_1 + 1, len(node_weights)):
                    input_pairs.append([input_1, input_2])

            assert len(input_pairs) == len(
                weights_pw_reduce[node_idx]
            ), "error: pairwise alpha length does not match pairwise terms length"

            keep_inputs = list()  # list of input nodes that are kept

            for input_pair_idx in range(len(input_pairs)):
                if (
                    weights_pw_reduce[node_idx][input_pair_idx] >= alpha_prune_threshold
                ):  # if pw weight larger than threshold keep input
                    keep_inputs.extend(input_pairs[input_pair_idx])

            weights_reduce_sparse.append(
                torch.stack(
                    [
                        (weight >= alpha_prune_threshold).type(torch.float)
                        if weight_idx in keep_inputs
                        else torch.zeros_like(weight)
                        for weight_idx, weight in enumerate(node_weights)
                    ]
                )
            )

        return weights_normal_sparse, weights_reduce_sparse

    def get_sparse_num_params(self, alpha_prune_threshold=0.0):
        """Get number of parameters for sparse one-shot-model

        Returns:
            A torch tensor
        """

        weights_normal, weights_reduce = self.get_sparse_alphas_pw(
            alpha_prune_threshold
        )
        # this returns tensors with only 0's and 1's depending on whether an op is used in the sparsified model

        # get none active ops/layer names

        # for normal cell
        none_active_ops_normal = list()
        for node_idx, node in enumerate(weights_normal):
            for mixed_op_idx, mixed_op in enumerate(node):
                none_active_ops_idx = (mixed_op == 0.0).nonzero()
                for op in none_active_ops_idx:
                    none_active_ops_normal.append(
                        str(node_idx)
                        + "."
                        + str(mixed_op_idx)
                        + "._ops."
                        + str(int(op))
                    )

        # and for reduction cell
        none_active_ops_reduce = list()
        for node_idx, node in enumerate(weights_reduce):
            for mixed_op_idx, mixed_op in enumerate(node):
                none_active_ops_idx = (mixed_op == 0.0).nonzero()
                for op in none_active_ops_idx:
                    none_active_ops_reduce.append(
                        str(node_idx)
                        + "."
                        + str(mixed_op_idx)
                        + "._ops."
                        + str(int(op))
                    )

        all_params = sum(
            p.numel() for p in self.net.parameters()
        )  # params of one-shot model

        # get normal and reduction layers
        normal_cells = list()
        red_cells = list()
        for lyr, cell in enumerate(self.net.cells):
            if cell.reduction:
                red_cells.append(lyr)
            else:
                normal_cells.append(lyr)

        # count params of non-active ops

        none_active_params = 0
        for layer_name, layer_weights in self.named_parameters():
            # check if layer is part of normal or reduction cell
            if "net.cells." in layer_name:  # layer part of cells at all?
                for cell in normal_cells:  # normal cell?
                    if "net.cells." + str(cell) in layer_name:  # normal cell
                        none_active_ops = none_active_ops_normal

                # else reduction cell
                for cell in red_cells:
                    if "net.cells." + str(cell) in layer_name:  # normal cell
                        none_active_ops = none_active_ops_reduce

                if any(
                    [none_active_op in layer_name for none_active_op in none_active_ops]
                ):  # check if layer is part of none-active ops
                    none_active_params += layer_weights.numel()

        active_params = all_params - none_active_params

        return active_params

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.net.modules():
            if isinstance(module, ops_naive.DropPath_):
                module.p = p

    def forward(self, x, sparsify_input_alphas=None):
        """Forward pass through the network

        Args:
            x: The input tensor
            sparsify_input_alphas: Whether to sparsify the alphas over the input nodes. Use `None`
                to not sparsify input alphas.
                For hierarchical alphas, `sparsify_input_alphas` should be a (float) threshold on
                the probability (i.e. between 0 and 1). Alphas above the threshold (and thus the
                corresponding input nodes) are kept.
                For pairwise alphas, if `sparsify_input_alphas` is larger than 0, then only the
                largest alpha is kept.
                Note that the sparsification is not be differentiable and thus cannot be used during
                training.

        Returns:
            The network output
        """

        (
            weights_normal,
            weights_reduce,
            weights_in_normal,
            weights_in_reduce,
            weights_pw_normal,
            weights_pw_reduce,
        ) = self._get_normalized_alphas()

        if len(self.device_ids) == 1:
            return self.net(
                x,
                weights_normal,
                weights_reduce,
                weights_in_normal,
                weights_in_reduce,
                weights_pw_normal,
                weights_pw_reduce,
                sparsify_input_alphas=sparsify_input_alphas,
                alpha_prune_threshold=self.alpha_prune_threshold,
            )

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        if weights_in_normal is not None:
            wnormal_in_copies = broadcast_list(weights_in_normal, self.device_ids)
            wreduce_in_copies = broadcast_list(weights_in_reduce, self.device_ids)
        else:
            wnormal_in_copies = None
            wreduce_in_copies = None

        if weights_pw_normal is not None:
            wnormal_pw_copies = broadcast_list(weights_pw_normal, self.device_ids)
            wreduce_pw_copies = broadcast_list(weights_pw_reduce, self.device_ids)
        else:
            wnormal_pw_copies = None
            wreduce_pw_copies = None

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(
            replicas,
            list(
                zip(
                    xs,
                    wnormal_copies,
                    wreduce_copies,
                    wnormal_in_copies,
                    wreduce_in_copies,
                    wnormal_pw_copies,
                    wreduce_pw_copies,
                )
            ),
            devices=self.device_ids,
        )
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        normalizer = self.get_normalizer(deterministic=True)
        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(normalizer(alpha))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(normalizer(alpha))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        if self.use_pairwise_input_alphas:

            weights_pw_normal = [
                F.softmax(alpha, dim=-1) for alpha in self.alpha_pw_normal
            ]
            weights_pw_reduce = [
                F.softmax(alpha, dim=-1) for alpha in self.alpha_pw_reduce
            ]

            gene_normal = gt.parse_pairwise(
                self.alpha_normal, weights_pw_normal, primitives=self.primitives
            )
            gene_reduce = gt.parse_pairwise(
                self.alpha_reduce, weights_pw_reduce, primitives=self.primitives
            )

        elif self.use_hierarchical_alphas:
            raise NotImplementedError
        else:

            gene_normal = gt.parse(self.alpha_normal, k=2, primitives=self.primitives)
            gene_reduce = gt.parse(self.alpha_reduce, k=2, primitives=self.primitives)

        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def named_weights_with_net(self):
        return self.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i : i + len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """

    def __init__(
        self,
        C_in,
        C,
        n_classes,
        n_layers,
        n_nodes=4,
        reduction_layers=[],
        stem_multiplier=3,
        PRIMITIVES=None,
        feature_scale_rate=2,
    ):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """

        assert PRIMITIVES is not None, "Error: need to define primitives"

        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False), nn.BatchNorm2d(C_cur)
        )

        if not reduction_layers:
            reduction_layers = [n_layers // 3, (2 * n_layers) // 3]

        print(f"Reduction layers: {reduction_layers}")

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in reduction_layers:
                C_cur *= feature_scale_rate
                reduction = True
            else:
                reduction = False

            cell = SearchCell(
                n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, PRIMITIVES
            )
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(
        self,
        x,
        weights_normal,
        weights_reduce,
        weights_in_normal=None,
        weights_in_reduce=None,
        weights_pw_normal=None,
        weights_pw_reduce=None,
        sparsify_input_alphas=None,
        alpha_prune_threshold=0.0,
    ):
        """Forward pass through the networks

        Args:
            x: The network input
            weights_normal: The alphas over operations for normal cells
            weights_reduce:The alphas over operations for reduction cells
            weights_in_normal: The alphas over inputs for normal cells (hierarchical alphas)
            weights_in_reduce: The alphas over inputs for reduction cells (hierarchical alphas)
            weights_pw_normal: The alphas over pairs of inputs for normal cells (pairwise alphas)
            weights_pw_reduce: The alphas over pairs of inputs for recution cells (pairwise alphas)
            sparsify_input_alphas: Whether to sparsify the alphas over the input nodes. Use `None`
                to not sparsify input alphas.
                For hierarchical alphas, `sparsify_input_alphas` should be a (float) threshold on
                the probability (i.e. between 0 and 1). Alphas above the threshold (and thus the
                corresponding input nodes) are kept.
                For pairwise alphas, if `sparsify_input_alphas` is larger than 0, then only the
                largest alpha is kept.
                Note that the sparsification is not be differentiable and thus cannot be used during
                training.

        Returns:
            The network output

        Note:
            Hierarchical and pairwise alphas are exclusive and only one of those can be specified
            (i.e. not None). Note that both hierarchical and pairwise alphas can be None.

        """
        s0 = s1 = self.stem(x)

        if sparsify_input_alphas:

            # always sparsify edge alphas (keep only edge with max prob for each previous node)
            weights_normal = sparsify_alphas(weights_normal)
            weights_reduce = sparsify_alphas(weights_reduce)

            if weights_in_normal is not None:
                weights_in_normal = sparsify_hierarchical_alphas(
                    weights_in_normal, sparsify_input_alphas
                )
                weights_in_reduce = sparsify_hierarchical_alphas(
                    weights_in_reduce, sparsify_input_alphas
                )
            elif weights_pw_normal is not None:
                weights_pw_normal = sparsify_pairwise_alphas(weights_pw_normal)
                weights_pw_reduce = sparsify_pairwise_alphas(weights_pw_reduce)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            weights_in = weights_in_reduce if cell.reduction else weights_in_normal
            weights_pw = weights_pw_reduce if cell.reduction else weights_pw_normal
            s0, s1 = s1, cell(
                s0,
                s1,
                weights,
                weights_in,
                weights_pw,
                alpha_prune_threshold=alpha_prune_threshold,
            )

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


def sparsify_alphas(w_input):
    """Sparsify regular (normalized) alphas

    Alphas are sparsified by keeping only the largest alpha for each previous node.

    Args:
        w_input: The list of alphas for each node in a cell

    Returns:
        The modified input alpha.
    """

    for idx in range(len(w_input)):
        max_val, primitive_indices = torch.topk(w_input[idx], 1)
        w_input[idx] = torch.zeros_like(w_input[idx])
        for row_idx in range(len(max_val)):
            w_input[idx][row_idx][primitive_indices[row_idx]] = max_val[row_idx]
    return w_input


def sparsify_hierarchical_alphas(w_input, threshold):  # deprecated
    """Sparsify hierarchical (normalized) alphas

    Alphas are sparsified by keeping only alphas above a threshold but at least the largest one

    Args:
        w_input: The list of hierarchical alphas for each node in a cell
        threshold: The threshold (between 0 and 1) above which input nodes are kept.

    Returns: The modified input list.

    Note:
        If there is no alpha above the threshold but multiple maximal values, then the last one is
        used (returned by `torch.max(t, 0)`)
    """
    for node_idx in range(len(w_input)):
        # w_node_in is of shape (num_input_nodes)x1
        if torch.any(w_input[node_idx] >= threshold):
            # sparsification yields at least one input node
            w_input[node_idx][w_input[node_idx] < threshold] = 0
        else:
            # no input with alpha above threshold -> keep largest one
            val, idx = torch.max(w_input[node_idx], 0)
            w_input[node_idx] = torch.zeros_like(w_input[node_idx])
            w_input[node_idx][idx] = val
    return w_input


def sparsify_pairwise_alphas(w_input):
    """Sparsify pairwise (normalized) alphas

    Alphas are sparsified by keeping only the largest alpha.

    Args:
        w_input: The list of pairwise alphas for each node in a cell

    Returns:
        The modified input list.

    Note:
        If there are multiple maximal values, then the last one is used (returned by
        `torch.max(t, 0)`)
    """
    for node_idx in range(len(w_input)):
        val, idx = torch.max(w_input[node_idx], 0)
        w_input[node_idx] = torch.zeros_like(w_input[node_idx])
        w_input[node_idx][idx] = val
    return w_input


class SearchCell(nn.Module):
    """Cell for searchs
    Each edge is mixed and continuous relaxed.

    Attributes:
        dag: List of lists where the out list corresponds to intermediate nodes in a cell. The inner
            list contains the mixed operations for each input node of an intermediate node (i.e.
            dag[i][j] calculates the outputs of the i-th intermediate node for its j-th input).
        preproc0: Preprocessing operation for the s0 input
        preproc1: Preprocessing operation for the s1 input
    """

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, PRIMITIVES):
        """
        Args:
            n_nodes: Number of intermediate n_nodes. The output of the cell is calculated by
                concatenating the outputs of all intermediate nodes in the cell.
            C_pp (int): C_out[k-2]
            C_p (int) : C_out[k-1]
            C (int)   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops_naive.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops_naive.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops_naive.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops_naive.MixedOp(C, stride, PRIMITIVES)
                self.dag[i].append(op)

    def forward(
        self, s0, s1, w_dag, w_input=None, w_pw=None, alpha_prune_threshold=0.0
    ):
        """Forward pass through the cell

        Args:
            s0: Output of the k-2 cell
            s1: Output of the k-1 cell
            w_dag: MixedOp weights ("alphas") (e.g. for n nodes and k primitive operations should be
                a list of length `n` of parameters where the n-th parameter has shape
                :math:`(n+2)xk = (number of inputs to the node) x (primitive operations)`)
            w_input: Distribution over inputs for each node (e.g. for n nodes should be a list of
                parameters of length `n`, where the n-th parameter has shape
                :math:`(n+2) = (number of inputs nodes)`).
            w_pw: weights on pairwise inputs for soft-pruning of inputs (e.g. for n nodes should be
                a list of parameters of length `n`, where the n-th parameter has shape
                :math:`(n+2) choose 2 = (number of combinations of two input nodes)`)
            alpha_prune_threshold:

        Returns:
            The output tensor of the cell
        """

        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]

        if w_input is not None:  # hierarchical alphas #####deprecated

            # iterate over nodes in cell
            for edges, w_node_ops, w_node_in in zip(self.dag, w_dag, w_input):
                s_cur = 0.0

                # iterate over inputs of node
                for i, (state_in, w_ops, w_in) in enumerate(
                    zip(states, w_node_ops, w_node_in)
                ):
                    if w_in > 0:
                        s_cur = s_cur + w_in * edges[i](state_in, w_ops)

                # equivalent but harder to read:
                # s_cur2 = sum(w2 * edges[i](s, w)
                #             for i, (s, w, w2) in enumerate(zip(states, w_node_ops, w_node_in)))
                # assert torch.allclose(s_cur2, s_cur)

                states.append(s_cur)

        elif w_pw is not None:  # pairwise alphas

            # iterate over nodes in cell
            for edges, w_node_ops, w_node_pw in zip(self.dag, w_dag, w_pw):
                pairwise_inputs = list()  # pairwise inputs
                unariy_inputs = list()  # unariy/single inputs

                # iterate over inputs per node
                for i, (state_in, w_ops) in enumerate(zip(states, w_node_ops)):

                    input_cur = edges[i](
                        state_in, w_ops, alpha_prune_threshold=alpha_prune_threshold
                    )
                    unariy_inputs.append(input_cur)

                # build pairwise sums
                for input_1 in range(len(unariy_inputs)):
                    for input_2 in range(input_1 + 1, len(unariy_inputs)):
                        pairwise_inputs.append(
                            unariy_inputs[input_1] + unariy_inputs[input_2]
                        )

                assert len(pairwise_inputs) == len(
                    w_node_pw
                ), "error: pairwise alpha length does not match pairwise terms length"

                s_cur = 0.0
                for i, sum_pw in enumerate(
                    pairwise_inputs
                ):  # weight pairwise sums by pw alpha
                    if w_node_pw[i] > alpha_prune_threshold:
                        s_cur = s_cur + sum_pw * w_node_pw[i]

                states.append(s_cur)

        else:  # regular darts

            for edges, w_list in zip(self.dag, w_dag):
                s_cur = sum(
                    edges[i](s, w, alpha_prune_threshold=alpha_prune_threshold)
                    for i, (s, w) in enumerate(zip(states, w_list))
                )

                states.append(s_cur)

        s_out = torch.cat(
            states[2:], dim=1
        )  # concatenate all intermediate nodes except inputs
        return s_out
