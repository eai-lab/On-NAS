""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
import utils.utils as utils
from utils import genotypes as gt
from models.search_cnn import SearchCNNController
from models.search_cnn_PC import SearchCNNControllerPC
from task_optimizer.darts import Darts,Architect
from task_optimizer.darts import train as d_train
import random
from tqdm import tqdm
import time
device = torch.device("cuda")

# tensorboard



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


def main(config):

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)
    _,_,_,_,test_data = utils.get_data(config.dataset, config.data_path, cutout_length=0, validation=True)



    # input my model architecture here
    normalizer = _init_alpha_normalizer(
        config.normalizer,
        config.task_train_steps,
        config.normalizer_t_max,
        config.normalizer_t_min,
        config.normalizer_temp_anneal_mode,
    )
    
    net_crit = nn.CrossEntropyLoss().to(device)
    
    model = SearchCNNController(           
            3,
            config.init_channels,
            config.k,
            config.layers,
            config,
            n_nodes=config.nodes,
            reduction_layers=config.reduction_layers,
            device_ids=config.gpus,
            normalizer=normalizer,
            PRIMITIVES=gt.PRIMITIVES,
            feature_scale_rate=1,
            use_hierarchical_alphas=config.use_hierarchical_alphas,
            use_pairwise_input_alphas=config.use_pairwise_input_alphas,
            alpha_prune_threshold=config.alpha_prune_threshold,
        )
    if config.meta_model == 'pc_adaptation':
            print("model created as PC adaptation")
            model = SearchCNNControllerPC(
                3,
                config.init_channels,
                config.k,
                config.layers,
                n_nodes=config.nodes,
                reduction_layers=config.reduction_layers,
                device_ids=config.gpus,
                normalizer=normalizer,
                PRIMITIVES=gt.PRIMITIVES,
                feature_scale_rate=1,
                use_hierarchical_alphas=config.use_hierarchical_alphas,
                use_pairwise_input_alphas=config.use_pairwise_input_alphas,
                use_pc_adaptation=True,
                alpha_prune_threshold=config.alpha_prune_threshold
            )
    

    ############################################################

    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.Adam(model.weights(), config.w_lr, betas=(0.0, 0.999),
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.0, 0.999),
                                   weight_decay=config.alpha_weight_decay)
    

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2 # changed here
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:]) #and order of these
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
   
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.workers,
                                              pin_memory=True)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, config.epochs, eta_min=0.0)
    architect = Architect(model, config.w_momentum, config.w_weight_decay, use_first_order_darts=True)

    # training loop
    best_top1 = 0.
    global_progress = 0
    from tqdm import tqdm 
    import pandas as pd 
    start_time = time.process_time()
    import copy

    warm_up_flag = False 
    epoch_avg = pd.DataFrame()
    for epoch in tqdm(range(config.epochs),total=config.epochs):
        mem = torch.cuda.memory_stats(0)['allocated_bytes.all.peak']/(1024**2)

        config.epoch_score = []
        lr = lr_scheduler.get_last_lr()[0]

        # training
        loader_chunk = Loader_Chunk(train_loader,valid_loader)
        
        if epoch < config.warm_up_epochs:
            warm_up_flag = True
            
            
        #a = list(self.parameters())[0].clone()
        # loss.backward()
        # self.optimizer.step()
        # b = list(self.parameters())[0].clone()
        # torch.equal(a.data, b.data)
        d_train(loader_chunk,model,architect,w_optim,alpha_optim,config.w_lr,global_progress,config,warm_up=warm_up_flag)
        
        # validation
        cur_step = (epoch+1) * len(train_loader)

        if epoch % 1 == 0: 
            val_switch(config,"before")
            top1 = validate(test_loader, model, epoch, cur_step, config)
            val_switch(config,"after")
            
            data = {"average":np.mean(config.epoch_score),"memory": mem}
        lr_scheduler.step()

        # log
        # genotype
        genotype = model.genotype()

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
   
        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
    end_time = time.process_time()-start_time
    times = pd.DataFrame([end_time])
    times.to_csv(f"./cifarsearch/{config.layers}_{config.exp_name}_non_trained_{config.sampleno}_sampled_warmup_{config.warm_up_epochs}_alpha_{config.alpha_expect}_lr_{config.w_lr}.csv",mode='a',index=False,header=False)

    torch.save(model.genotype(),f"genotype_{config.exp_name}.pt")
    
    

def val_switch(config,b_or_f):
    if b_or_f == "before":
        config.naivenaive = 1
        config.eval_switch = 1
        config.cell_phase = 3
    else:
        config.naivenaive = 0 
        config.eval_switch = 0
        config.cell_phase = 3 
        

def validate(valid_loader, model, epoch, cur_step, config):
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    
    model.eval()
    
    test_p_bar_during = tqdm(enumerate(valid_loader),total=len(valid_loader))
    with torch.no_grad():
        for step, (X, y) in test_p_bar_during:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, config, topk=(1, 5))
            test_p_bar_during.set_postfix({'Prec1': prec1.item(), 'Prec5': prec5.item(), 'losses' : loss.item()})

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

    return top1.avg

class Loader_Chunk():
    def __init__(self,tl,vl):
        self.train_loader = tl
        self.valid_loader = vl 



def _str_or_none(x):
    """Convert multiple possible input strings to None"""
    return None if (x is None or not x or x.capitalize() == "None") else x

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Search Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution
    parser.add_argument("--name", required=True)
    parser.add_argument("--job_id", default=None, type=_str_or_none)
    parser.add_argument("--path", default="/home/elt4hi/")
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
        default=100,
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
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--w_lr", type=float, default=0.025, help="lr for weights") #0.025 was the default #0.01 was one from darts paper
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
    parser.add_argument("--layers", type=int, default=5, help="# of layers (cells)")
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
        help="we cannot run exp runs all time..."
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
        default=1,
        help="exp for lighter darts"

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
        help="grad from previous phase"
        
    )
    parser.add_argument(
        "--alpha_previous_grad",
        default=None,
        help="grad from previous phase"
        
    )
    
    parser.add_argument(
        "--alpha_pprevious_grad",
        default=None,
        help="grad from previous phase"
        
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
        default=1,
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
        help="how much chunks we want to make? "
    )

    parser.add_argument(
        "--loss_track",
        type = list,
        default= [],
        help="tracking losses, are they learning? "
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
        "--epochs",
        type = int,
        default= 0,
        help="Testing Epochs... "
    )
    parser.add_argument(
        "--epoch_score",
        type = list,
        default= [],
        help="To test acc... "
    )
    parser.add_argument(
        "--exp_name",
        default= "no_name",
        help="To name csvs... "
    )
    parser.add_argument(
        "--stackup_phase",
        default = 0,
        type=int
        
    )
    parser.add_argument(
        "--step_by_step",
        default = 0,
        type=int
        
    )
    parser.add_argument(
        "--prohibited_list",
        default = ["cifar10","cifar100","colored_mnist"],
        type=list
        
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
    main(args)
