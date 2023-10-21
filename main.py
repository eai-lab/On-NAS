""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""
"""

"""

## Packages 

# =======================================

import argparse
from collections import OrderedDict
import copy
import os
import time
import numpy as np
import pickle
import torch
#import wandb
from meta_optimizer.reptile import NAS_Reptile
from models.search_cnn import SearchCNNController, SearchCell, SearchCNN
from models.search_cnn_PC import SearchCNNControllerPC
from models.search_cnn_pure_darts import SearchCNNController_puredarts
from task_optimizer.darts import Darts
from utils import genotypes as gt
from utils import utils
from tqdm import tqdm
import torch.nn as nn
import random



def TFML(config, task_optimizer_cls = Darts, meta_optimizer_cls = NAS_Reptile ):
    
    config.logger.info("Two-Fold Meta-learning, initializing... ")
    
    
    # Randomness Control
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
     # hyperparameter settings
    if config.use_hp_setting:
        config = utils.set_hyperparameter(config)
    else:
        print("Not using hp_setting.")

    if config.use_torchmeta_loader:
        from tasks.torchmeta_loader import (
            OmniglotFewShot,
            MiniImageNetFewShot as miniImageNetFewShot,
        )
    else:
        raise RuntimeError(f"Other data loaders deprecated.")

    if config.dataset == "omniglot":
        task_distribution_class = OmniglotFewShot
    elif config.dataset == "miniimagenet":
        task_distribution_class = miniImageNetFewShot

    else:
        raise RuntimeError(f"Dataset {config.dataset} is not supported.")

    # task distribution
    task_distribution = task_distribution_class(config, download=True)            
    
    # meta model 
    normalizer = _init_alpha_normalizer(
        config.normalizer,
        config.task_train_steps,
        config.normalizer_t_max,
        config.normalizer_t_min,
        config.normalizer_temp_anneal_mode,
    )
    
    meta_model = _build_model(config, task_distribution, normalizer)
    
    meta_cell = copy.deepcopy(meta_model.net.cells[1])
    n_param = param_count(meta_model.weights())
    
    config, meta_optimizer = _init_meta_optimizer(
        config, meta_optimizer_cls, meta_model, meta_cell
    )
    
    config, task_optimizer = _init_task_optimizer(
        config, task_optimizer_cls, meta_model
    )
    config.logger = logger

    train_info = dict()
    
    if config.light_exp:
        if config.naivenaive != 1:
            from meta_optimizer.reptile import alter_weights
            saved_cell = meta_model.net.cells[1]
            # if config.meta_model == "pc_adaptation":
            #     print("Loading pc adaptation model")
            #     meta_model.load_state_dict(torch.load('{pc model path}}',map_location="cuda:0"))
            # else:
            #     print("Loading ours model")
            #     meta_model.load_state_dict(torch.load('{model path}',map_location="cuda:0"))
            # saved_cell.load_state_dict(torch.load('{cell path}',map_location="cuda:0"))
            # meta_model.net.wocelllist.load_state_dict(torch.load('{misc path}'))
            # alter_weights(meta_model,saved_cell)

            print("weights altered with trained cells")

        config, alpha_logger, sparse_params = onnas_eval(config, meta_model ,task_distribution, task_optimizer)
        print("Finished")
        return
        
        

    

    if config.eval != 1 and config.light_exp != 1:
        config, meta_model, train_info = train(
            config,
            meta_cell,
            meta_model,
            task_distribution,
            task_optimizer,
            meta_optimizer,
            train_info,
        )
    
    # save results
    experiment = {
        "meta_genotype": meta_model.genotype(),
        "alphas": [alpha for alpha in meta_model.alphas()],
        "final_eval_test_accu": config.top1_logger_test.avg,
        "final_eval_test_loss": config.losses_logger_test.avg,
    }
    experiment.update(train_info)
    prefix = config.eval_prefix
    pickle_to_file(experiment, os.path.join(config.path, prefix + "experiment.pickle"))
    pickle_to_file(config, os.path.join(config.path, prefix + "config.pickle"))
    print("Finished")




def param_count(p_gen):
    return sum(p.numel() for p in p_gen if p.requires_grad)


def _init_meta_optimizer(config, meta_optimizer_class, meta_model,meta_cell):
    if meta_optimizer_class == NAS_Reptile:
        # reptile uses SGD as meta optim
        config.w_meta_optim = torch.optim.SGD(meta_model.weights(), lr=config.w_meta_lr)

        if meta_model.alphas() is not None:
            config.a_meta_optim = torch.optim.SGD(
                meta_model.alphas(), lr=config.a_meta_lr
            )
        else:
            config.a_meta_optim = None
    else:
        raise RuntimeError(
            f"Meta-Optimizer {meta_optimizer_class} is not yet supported."
        )
    meta_optimizer = meta_optimizer_class(meta_model, meta_cell, config)
    return config, meta_optimizer



def _init_alpha_normalizer(name, task_train_steps, t_max, t_min, temp_anneal_mode):
    normalizer = dict()
    normalizer["name"] = name
    normalizer["params"] = dict()
    normalizer["params"]["curr_step"] = 0.0  # current step for scheduling normalizer
    normalizer["params"]["max_steps"] = float(
        task_train_steps
    )  # for scheduling normalizer
    normalizer["params"]["t_max"] = t_max
    normalizer["params"]["t_min"] = t_min
    normalizer["params"]["temp_anneal_mode"] = temp_anneal_mode  # temperature annealing
    return normalizer


def _build_model(config, task_distribution, normalizer):
    if config.meta_model == "searchcnn":
        meta_model = SearchCNNController(
            task_distribution.n_input_channels,
            config.init_channels,
            task_distribution.n_classes,
            config.layers,
            config,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            normalizer=normalizer,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
            feature_scale_rate=1,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            alpha_prune_threshold=config.alpha_prune_threshold,
        )
        
    elif config.meta_model == "pure_darts":
        meta_model = SearchCNNController_puredarts(
            task_distribution.n_input_channels,
            config.init_channels,
            task_distribution.n_classes,
            config.layers,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            normalizer=normalizer,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
            feature_scale_rate=1,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            alpha_prune_threshold=config.alpha_prune_threshold,
        )
    

    
        
    elif config.meta_model == 'pc_adaptation':
        meta_model = SearchCNNControllerPC(
            task_distribution.n_input_channels,
            config.init_channels,
            task_distribution.n_classes,
            config.layers,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            normalizer=normalizer,
            PRIMITIVES=gt.PRIMITIVES_FEWSHOT,
            feature_scale_rate=1,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            use_pc_adaptation=True,
            alpha_prune_threshold=config.alpha_prune_threshold,
        ) 
   
    else:
        raise RuntimeError(f"Unknown meta_model {config.meta_model}")
   
    return meta_model.to(config.device)


def _init_task_optimizer(config, task_optimizer_class, meta_model):
    return config, task_optimizer_class(meta_model, config)



def train(
    config,
    meta_cell,
    meta_model,
    task_distribution,
    task_optimizer,
    meta_optimizer,
    train_info=None,
):
    """Meta-training loop

    Args:
        config: Training configuration parameters
        meta_cell : The meta_cell
        meta_model: The meta_model
        task_distribution: Task distribution object
        task_optimizer: A pytorch optimizer for task training
        meta_optimizer: A pytorch optimizer for meta training
        train_info: Dictionary that is added to the experiment.pickle file in addition to training
            internal data.

    Returns:
        A tuple containing the updated config, meta_model and updated train_info.
    """
    if train_info is None:
        train_info = dict()
    else:
        assert isinstance(train_info, dict)

    # add training performance to train_info
    train_test_loss = list()
    train_test_accu = list()
    test_test_loss = list()
    test_test_accu = list()
    train_info["train_test_loss"] = train_test_loss
    train_info["train_test_accu"] = train_test_accu
    train_info["test_test_loss"] = test_test_loss
    train_info["test_test_accu"] = test_test_accu

    # time averages for logging (are reset during evaluation)
    io_time = utils.AverageMeter()
    sample_time = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    total_time = utils.AverageMeter()

    # performance logger
    config.top1_logger = utils.AverageMeter()
    config.top1_logger_test = utils.AverageMeter()
    config.losses_logger = utils.AverageMeter()
    config.losses_logger_test = utils.AverageMeter()

    # meta lr annealing
    w_meta_lr_scheduler, a_meta_lr_scheduler = _get_meta_lr_scheduler(
        config, meta_optimizer
    )
    normalizer = _init_alpha_normalizer(
        config.normalizer,
        config.task_train_steps,
        config.normalizer_t_max,
        config.normalizer_t_min,
        config.normalizer_temp_anneal_mode,
    )
    if config.wandb:
        wandb.init(config=config,name=config.wandb_name)


    
    print(config.logger)
    
    for meta_epoch in tqdm(range(config.start_epoch, config.meta_epochs + 1)):
        
            

        time_es = time.time()
        meta_train_batch = task_distribution.sample_meta_train()
        time_samp = time.time()

        sample_time.update(time_samp - time_es)

        # Each task starts with the current meta state
        meta_state = copy.deepcopy(meta_model.state_dict())
        global_progress = f"[Meta-Epoch {meta_epoch:2d}/{config.meta_epochs}]"
        task_infos = []
        time_bs = time.time()
        
        for task in meta_train_batch:
            
            current_task_info = task_optimizer.step(
                    task, epoch=meta_epoch, global_progress=global_progress
            )
            task_infos += [current_task_info]
            meta_model.load_state_dict(meta_state)
            

        time_be = time.time()

        batch_time.update(time_be - time_bs)

        train_test_loss.append(config.losses_logger.avg)
        train_test_accu.append(config.top1_logger.avg)
        # do a meta update
        # make an insurance
        
        
        meta_optimizer.step(task_infos,meta_cell)
                    


                # meta model has been updated.

                # update meta LR
        if (a_meta_lr_scheduler is not None) and (meta_epoch >= config.warm_up_epochs):
            a_meta_lr_scheduler.step()


        if w_meta_lr_scheduler is not None:
            w_meta_lr_scheduler.step()

        time_ee = time.time()
        total_time.update(time_ee - time_es)

        if meta_epoch % config.print_freq == 0:
            config.logger.info(
                f"Train: [{meta_epoch:2d}/{config.meta_epochs}] "
                f"Time (sample, batch, sp_io, total): {sample_time.avg:.2f}, {batch_time.avg:.2f}, "
                f"{io_time.avg:.2f}, {total_time.avg:.2f} "
                f"Train-TestLoss {config.losses_logger.avg:.3f} "
                f"Train-TestPrec@(1,) ({config.top1_logger.avg:.1%}, {1.00:.1%})"
            )

        
        # meta testing every config.eval_freq epochs
        if meta_epoch % config.eval_freq == 0:  # meta test eval + backup
            meta_test_batch = task_distribution.sample_meta_test()
            #this is the place where you changed

            # Each task starts with the current meta state
            meta_state = copy.deepcopy(meta_model.state_dict())
            # copy also the optimizer states
            meta_optims_state = [
                copy.deepcopy(meta_optimizer.w_meta_optim.state_dict()),
                copy.deepcopy(meta_optimizer.a_meta_optim.state_dict()),
                copy.deepcopy(task_optimizer.w_optim.state_dict()),
                copy.deepcopy(task_optimizer.a_optim.state_dict()),
                copy.deepcopy(meta_optimizer.w_meta_cell_optim.state_dict()),
            ]
            
            global_progress = f"[Meta-Epoch {meta_epoch:2d}/{config.meta_epochs}]"
            task_infos = []
            for task in meta_test_batch:

                task_infos += [
                    task_optimizer.step(
                        task,
                        epoch=meta_epoch,
                        global_progress=global_progress,
                        test_phase=True,
                    )
                ]
                meta_model.load_state_dict(meta_state)
            config.logger.info(
                f"Train: [{meta_epoch:2d}/{config.meta_epochs}] "
                f"Test-TestLoss {config.losses_logger_test.avg:.3f} "
                f"Test-TestPrec@(1,) ({config.top1_logger_test.avg:.1%}, {1.00:.1%})"
            )
           
       
            test_test_loss.append(config.losses_logger_test.avg)
            test_test_accu.append(config.top1_logger_test.avg)

            # print cells
            config.logger.info(f"genotype = {task_infos[0].genotype}")
            config.logger.info(
                f"alpha vals = {[alpha for alpha in meta_model.alphas()]}"
            )

            # reset the states so that meta training doesnt see meta testing
            meta_optimizer.w_meta_optim.load_state_dict(meta_optims_state[0])
            meta_optimizer.a_meta_optim.load_state_dict(meta_optims_state[1])
            task_optimizer.w_optim.load_state_dict(meta_optims_state[2])
            task_optimizer.a_optim.load_state_dict(meta_optims_state[3])
            meta_optimizer.w_meta_cell_optim.load_state_dict(meta_optims_state[4])

            # save checkpoint
            experiment = {
                "genotype": [task_info.genotype for task_info in task_infos],
                "meta_genotype": meta_model.genotype(),
                "alphas": [alpha for alpha in meta_model.alphas()],
            }
            experiment.update(train_info)
            pickle_to_file(experiment, os.path.join(config.path, "experiment.pickle"))

            utils.save_state(
                meta_model,
                meta_optimizer,
                task_optimizer,
                config.path,
                meta_epoch,
                job_id=config.job_id,
            )

            # reset time averages during testing
            sample_time.reset()
            batch_time.reset()
            total_time.reset()
            io_time.reset()

    torch.save(meta_cell, os.path.join(config.path, "meta_cell.pt"))

    # end of meta train
    utils.save_state(
        meta_model, meta_optimizer, task_optimizer, config.path, job_id=config.job_id
    )
    experiment = {
        "meta_genotype": meta_model.genotype(),
        "alphas": [alpha for alpha in meta_model.alphas()],
    }
    experiment.update(train_info)
    pickle_to_file(experiment, os.path.join(config.path, "experiment.pickle"))
    pickle_to_file(config, os.path.join(config.path, "config.pickle"))

    return config, meta_model, train_info



def pickle_to_file(var, file_path):
    """Save a single variable to a file using pickle"""
    with open(file_path, "wb") as handle:
        pickle.dump(var, handle)
        
        

def _get_meta_lr_scheduler(config, meta_optimizer):
    if config.w_meta_anneal:
        w_meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer.w_meta_optim, config.meta_epochs, eta_min=0.0
        )

        if w_meta_lr_scheduler.last_epoch == -1:
            w_meta_lr_scheduler.step()
    else:
        w_meta_lr_scheduler = None

    if config.a_meta_anneal and config.a_meta_optim is not None:
        a_meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer.a_meta_optim,
            (config.meta_epochs - config.warm_up_epochs),
            eta_min=0.0,
        )

        if a_meta_lr_scheduler.last_epoch == -1:
            a_meta_lr_scheduler.step()
    else:
        a_meta_lr_scheduler = None

    return w_meta_lr_scheduler, a_meta_lr_scheduler

def onnas_eval(config, meta_model, task_distribution, task_optimizer):
    """Meta-testing

    Returns:
        A tuple consisting of (config, alpha_logger).
        The config contains the fields `top1_logger_test` with the average top1 accuracy and
        `losses_logger_test` with the average loss during meta test test.
        The alpha logger contains lists of architecture alpha parameters.
    """
    # Each task starts with the current meta state, make a backup
    meta_state = copy.deepcopy(meta_model.state_dict())
    # copy also the task optimizer states
    meta_optims_state = [
        copy.deepcopy(task_optimizer.w_optim.state_dict()),
        copy.deepcopy(task_optimizer.a_optim.state_dict()),
    ]

    top1_test = utils.AverageMeter()
    losses_test = utils.AverageMeter()
    config.top1_logger_test = top1_test
    config.losses_logger_test = losses_test
    paramas_logger = list()

    if config.meta_model == "searchcnn":
        alpha_logger = OrderedDict()
        alpha_logger["normal_relaxed"] = list()
        alpha_logger["reduced_relaxed"] = list()
        alpha_logger["genotype"] = list()
        alpha_logger["all_alphas"] = list()
        alpha_logger["normal_hierarchical"] = list()
        alpha_logger["reduced_hierarchical"] = list()
        alpha_logger["normal_pairwise"] = list()
        alpha_logger["reduced_pairwise"] = list()
    else:
        alpha_logger = None



    
    for eval_epoch in range(config.eval_epochs):
        config.computing_time = 0
        meta_test_batch = task_distribution.sample_meta_test()
        
        global_progress = f"[Eval-Epoch {eval_epoch:2d}/{config.eval_epochs}]"
        task_infos = []
 
        time_ts = time.time()
        for task in (meta_test_batch):
            task_infos += [
                task_optimizer.step(
                    task,
                    epoch=config.meta_epochs,
                    global_progress=global_progress,
                    test_phase=True,
                    alpha_logger=alpha_logger,
                    sparsify_input_alphas=config.sparsify_input_alphas,
                )
            ]
        time_te = time.time()
        # load meta state
        meta_model.load_state_dict(meta_state)
        task_optimizer.w_optim.load_state_dict(meta_optims_state[0])
        task_optimizer.a_optim.load_state_dict(meta_optims_state[1])

        

        

    return config, alpha_logger, paramas_logger


def evaluate(config, meta_model, task_distribution, task_optimizer):
    """Meta-testing

    Returns:
        A tuple consisting of (config, alpha_logger).
        The config contains the fields `top1_logger_test` with the average top1 accuracy and
        `losses_logger_test` with the average loss during meta test test.
        The alpha logger contains lists of architecture alpha parameters.
    """
    # Each task starts with the current meta state, make a backup
    meta_state = copy.deepcopy(meta_model.state_dict())
    # copy also the task optimizer states
    meta_optims_state = [
        copy.deepcopy(task_optimizer.w_optim.state_dict()),
        copy.deepcopy(task_optimizer.a_optim.state_dict()),
    ]

    top1_test = utils.AverageMeter()
    losses_test = utils.AverageMeter()
    config.top1_logger_test = top1_test
    config.losses_logger_test = losses_test
    paramas_logger = list()

    if config.meta_model == "searchcnn":
        alpha_logger = OrderedDict()
        alpha_logger["normal_relaxed"] = list()
        alpha_logger["reduced_relaxed"] = list()
        alpha_logger["genotype"] = list()
        alpha_logger["all_alphas"] = list()
        alpha_logger["normal_hierarchical"] = list()
        alpha_logger["reduced_hierarchical"] = list()
        alpha_logger["normal_pairwise"] = list()
        alpha_logger["reduced_pairwise"] = list()
    else:
        alpha_logger = None

    


    for eval_epoch in range(config.eval_epochs):

        meta_test_batch = task_distribution.sample_meta_test()

        global_progress = f"[Eval-Epoch {eval_epoch:2d}/{config.eval_epochs}]"
        task_infos = []

        for task in (meta_test_batch):
            time_ts = time.time()
            task_infos += [
                task_optimizer.step(
                    task,
                    epoch=config.meta_epochs,
                    global_progress=global_progress,
                    test_phase=True,
                    alpha_logger=alpha_logger,
                    sparsify_input_alphas=config.sparsify_input_alphas,
                )
            ]
            time_te = time.time()
            # load meta state
            meta_model.load_state_dict(meta_state)

            task_optimizer.w_optim.load_state_dict(meta_optims_state[0])
            task_optimizer.a_optim.load_state_dict(meta_optims_state[1])

            if isinstance(meta_model, SearchCNNController):
                paramas_logger.append(task_infos[-1].sparse_num_params)
            else:
                paramas_logger.append(utils.count_params(meta_model))

        prefix = f" (prefix: {config.eval_prefix})" if config.eval_prefix else ""

        config.logger.info(
            f"Test data evaluation{prefix}:: [{eval_epoch:2d}/{config.eval_epochs}] "
            f"Test-TestLoss {config.losses_logger_test.avg:.3f} "
            f"Test-TestPrec@(1,) ({config.top1_logger_test.avg:.1%}, {1.00:.1%})"
            f" \n Sparse_num_params (mean, min, max): {np.mean(paramas_logger)}, {np.min(paramas_logger)}, {np.max(paramas_logger)}"
        )
    
    return config, alpha_logger, paramas_logger


def _str_or_none(x):
    """Convert multiple possible input strings to None"""
    return None if (x is None or not x or x.capitalize() == "None") else x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Search Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution
    parser.add_argument("--name", required=True)
    parser.add_argument("--job_id", default=None, type=_str_or_none)
    parser.add_argument("--path", default="{DEFAULT PATH HERE}")
    parser.add_argument("--data_path")
    parser.add_argument("--seed", type=int, default=21, help="random seed")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument(
        "--eval_prefix",
        type=str,
        default="",
        help="Prefix added to all output files during evaluation",
    ) 

    # only for hp search
    parser.add_argument(
        "--hp_setting", type=str, default="in", help="use predefined HP configuration"
    )
    parser.add_argument("--use_hp_setting", type=int, default=0)

    ################################
    parser.add_argument("--workers", type=int, default=4, help="# of workers")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument(
        "--use_torchmeta_loader",
        action="store_true",
        help="Use torchmeta for data loading.",
    )
    parser.add_argument("--dataset", default="omniglot", help="omniglot / miniimagenet")
    parser.add_argument(
        "--use_vinyals_split",
        action="store_true",
        help="Only relevant for Omniglot: Use the vinyals split. Requires the "
        "torchmeta data loading.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpu device ids separated by comma. " "`all` indicates use all gpus.",
    )

    # Meta Learning
    parser.add_argument(
        "--meta_model", type=str, default="searchcnn", help="meta model to use"
    )
    parser.add_argument("--model_path", default=None, help="load model from path")

    parser.add_argument(
        "--meta_epochs", type=int, default=10, help="Number meta train epochs"
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Start training at a specific epoch (for resuming training from a checkpoint)",
    )
    parser.add_argument(
        "--meta_batch_size", type=int, default=5, help="Number of tasks in a meta batch"
    )
    parser.add_argument(
        "--test_meta_batch_size",
        type=int,
        default=1,
        help="Number of tasks in a test meta batch",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=100,
        help="Number of epochs for final evaluation of test data",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="how often to run meta-testing for intermediate evaluation (in epochs)",
    )

    parser.add_argument(
        "--task_train_steps",
        type=int,
        default=1,
        help="Number of training steps per task",
    )
    parser.add_argument(
        "--test_task_train_steps",
        type=int,
        default=1,
        help="Number of training steps per task",
    )

    parser.add_argument(
        "--warm_up_epochs",
        type=int,
        default=1e6,
        help="warm up epochs before architecture search is enabled",
    )

    parser.add_argument(
        "--test_adapt_steps",
        type=float,
        default=1.0,
        help="for how many test-train steps should architectue be adapted (relative to test_train_steps)?",
    )

    parser.add_argument(
        "--w_meta_optim", default=None, help="Meta optimizer of weights"
    )
    parser.add_argument(
        "--w_meta_lr", type=float, default=0.001, help="meta lr for weights"
    )
    parser.add_argument(
        "--w_meta_anneal", type=int, default=1, help="Anneal Meta weights optimizer LR"
    )
    parser.add_argument(
        "--w_task_anneal", type=int, default=0, help="Anneal task weights optimizer LR"
    )
    parser.add_argument("--a_meta_optim", default=None, help="Meta optimizer of alphas")
    parser.add_argument(
        "--a_meta_lr", type=float, default=0.001, help="meta lr for alphas"
    )
    parser.add_argument(
        "--a_meta_anneal",
        type=int,
        default=1,
        help="Anneal Meta architecture optimizer LR",
    )
    parser.add_argument(
        "--a_task_anneal",
        type=int,
        default=0,
        help="Anneal task architecture optimizer LR",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="softmax",
        help="Alpha normalizer",
        choices=["softmax", "relusoftmax", "gumbel_softmax"],
    )
    parser.add_argument(
        "--normalizer_temp_anneal_mode",
        type=str,
        default=None,
        help="Temperature anneal mode (if applicable to normalizer)",
    )
    parser.add_argument(
        "--normalizer_t_max", type=float, default=5.0, help="Initial temperature"
    )
    parser.add_argument(
        "--normalizer_t_min",
        type=float,
        default=0.1,
        help="Final temperature after task_train_steps",
    )

    # Few Shot Learning
    parser.add_argument(
        "--n",
        default=1,
        type=int,
        help="Training examples per class / support set (for meta testing).",
    )
    
    parser.add_argument(
        "--n_train",
        default=15,
        type=int,
        help="Training examples per class for meta training.",
    )

    parser.add_argument("--k", default=5, type=int, help="Number of classes.")
    parser.add_argument(
        "--q", default=1, type=int, help="Test examples per class / query set"
    )

    # Weights
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--w_lr", type=float, default=0.025, help="lr for weights")
    parser.add_argument(
        "--w_momentum", type=float, default=0.0, help="momentum for weights"
    )
    parser.add_argument(
        "--w_weight_decay", type=float, default=0.0, help="weight decay for weights"
    )
    parser.add_argument(
        "--w_grad_clip", type=float, default=10e5, help="gradient clipping for weights"
    )

    parser.add_argument(
        "--drop_path_prob", type=float, default=0.0, help="drop path probability"
    )
    parser.add_argument(
        "--use_drop_path_in_meta_testing",
        action="store_true",
        help="Whether to use drop path also during meta testing.",
    )

    # Architectures
    parser.add_argument("--init_channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4, help="# of layers (cells)")
    parser.add_argument("--nodes", type=int, default=3, help="# of nodes per cell") 
    parser.add_argument(
        "--use_hierarchical_alphas",
        action="store_true",
        help="Whether to use hierarhical alphas in search_cnn model.",
    )
    parser.add_argument(
        "--use_pairwise_input_alphas",
        action="store_true",
        help="Whether to use alphas on pairwise inputs in search_cnn model.",
    )
    parser.add_argument(
        "--reduction_layers",
        nargs="+",
        default=[],
        type=int,
        help="Where to use reduction cell",
    )
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="lr for alpha")
    parser.add_argument(
        "--alpha_prune_threshold",
        type=float,
        default=0.0,
        help="During forward pass, alphas below the threshold probability are pruned (meaning "
        "the respective operations are not executed anymore).",
    )
    parser.add_argument(
        "--meta_model_prune_threshold",
        type=float,
        default=0.0,
        help="During meta training, prune alphas from meta model below this threshold to not train them any         longer.",
    )

    parser.add_argument(
        "--alpha_weight_decay", type=float, default=0.001, help="weight decay for alpha"
    )
    parser.add_argument(
        "--anneal_softmax_temperature",
        action="store_true",
        help="anneal temperature of softmax",
    )
    parser.add_argument(
        "--do_unrolled_architecture_steps",
        action="store_true",
        help="do one step in w before computing grad of alpha",
    )

    parser.add_argument(
        "--use_first_order_darts",
        action="store_true",
        help="Whether to use first order DARTS.",
    )

    parser.add_argument(
        "--sparsify_input_alphas",
        type=float,
        default=None,
        help="sparsify_input_alphas input for the search_cnn forward pass "
        "during final evaluation.",
    )  #### deprecated
    # Experiment
    parser.add_argument(
        "--exp_const",
        type=int,
        default=0,
        help="begin experiment or not"
    )
    parser.add_argument(
        "--exp_cell",
        type=int,
        default = 0,
        help="cell experiments, begin"
    )
    parser.add_argument(
        "--wandb",
        type=int,
        default=0,
        help=""
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default='lab_pc',
        help="define name of the run"
    )
    
    parser.add_argument(
        "--const_mult",
        type=int,
        default=1,
        help="define mult of task const"
    )
    parser.add_argument(
        "--unittest",
        type=int,
        default=0,
        help="cell unit test"
    )
    parser.add_argument(
        "--cell_const_mult",
        type=int,
        default=4,
        help="cell const mult."
    )
    parser.add_argument(
        "--cell_const_flag",
        type=int,
        default=1,
        help="cell const flag."
    )
    parser.add_argument(
        "--light_exp",
        type=int,
        default=0,
        help="exp for onnas"
    )
    parser.add_argument(
        "--opsample",
        type=int,
        default=1,
        help="exp for ops_sampling"
    )
    parser.add_argument(
        "--sampleno",
        type=int,
        default=7,
        help="exp for ops_sampling"
    )
    parser.add_argument(
        "--naivenaive",
        type=int,
        default=0,
        help="exp for naive full model"
    )
    parser.add_argument(
        "--cell_phase",
        type=int,
        default=2,
        help="cell_phase"
    )
    parser.add_argument(
        "--cell_idx",
        type=int,
        default=0,
        help="cell_phase"
    )
    parser.add_argument(
        "--previous_grad",
        default=None,
        help="grad from previous phase"
    )
    
    parser.add_argument(
        "--pprevious_grad",
        default=None,
        help="grad from previous previous phase"
    )
    parser.add_argument(
        "--alpha_previous_grad",
        default=None,
        help="grad from previous phase"
    )
    
    parser.add_argument(
        "--alpha_pprevious_grad",
        default=None,
        help="grad from previous previous phase"
    )
    
    parser.add_argument(
        "--eval_switch",
        default=0,
        type=int,
        help="switch"
    )
    parser.add_argument(
        "--residual_flag",
        type=int,
        default=0,
        help="switch of residual"
    )

    parser.add_argument(
        "--beta_sampling",
        type=int,
        default=0,
        help="switch of betasampling"
    )
    parser.add_argument(
        "--alpha_grad_footprints",
        default=list(),
        help="I record n-alpha-grads"
    )
    parser.add_argument(
        "--alpha_sample_metrics",
        default=None,
        help="I record the alpha metrics"
    )
    parser.add_argument(
        "--total_steps",
        type = int,
        default=0,
        help="total steps we are ongoing "
    )
    parser.add_argument(
        "--alpha_expect",
        type = int,
        default=0,
        help="flag wheter we expect or not "
    )
    parser.add_argument(
        "--grad_acc",
        type = int,
        default=None,
       
    )
    parser.add_argument(
        "--split_num",
        type = int,
        default=1,
        help="how much chunks we want to make for the gradient accumulation? "
    )
    parser.add_argument(
        "--loss_track",
        type = list,
        default= [],
        help="tracking losses, are they learning? "
    )
    parser.add_argument(
        "--loss_val_track",
        type = list,
        default= [],
        help="tracking val losses, are they learning? "
    )
    parser.add_argument(
        "--computing_time",
        type = float,
        default= 0.0,
        help="Time for cuda computing "
    )
    parser.add_argument(
        "--naivesample",
        type = int,
        default= 0,
        help="Sampling Naives... "
    )
    parser.add_argument(
        "--memory_snap",
        type = float,
        default = 0.0,
        help="to get precise snap of memory."
    )
    parser.add_argument(
        "--exp_name",
        type = str,
        default = None,
       
    )
    parser.add_argument(
        "--step_by_step",
        type = int,
        default = 0,
    )
    parser.add_argument(
        "--stackup_phase",
        type = int,
        default = 0,
    )
    parser.add_argument(
        "--prohibited_list",
        type=list,
        default=[]
    )
       
    
    args = parser.parse_args()

    args.path = os.path.join(
        args.path, ""
    )  # add file separator at end if it does not exist
    args.plot_path = os.path.join(args.path, "plots")

    # Setup data and hardware config
    # config.data_path = "datafiles"
    args.gpus = utils.parse_gpus(args.gpus)
    args.device = torch.device("cuda")

    # Logging
    logger = utils.get_logger(os.path.join(args.path, f"{args.name}.log"))
    args.logger = logger
    TFML(args)  
