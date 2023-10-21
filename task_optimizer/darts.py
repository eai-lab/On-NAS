""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""
from tqdm import tqdm
import copy
import time
import torch
import torch.nn as nn
from collections import OrderedDict, namedtuple
from meta_optimizer.reptile import get_finite_difference
from utils import utils
from models.search_cnn import SearchCNNController


          

class Darts:
    def __init__(self, model, config, do_schedule_lr=False):

        self.config = config
        self.config.logger = None
        self.model = model
        self.do_schedule_lr = do_schedule_lr
        self.task_train_steps = config.task_train_steps
        self.test_task_train_steps = config.test_task_train_steps
        self.warm_up_epochs = config.warm_up_epochs
        self.eval_switch = 0
        self.pprevious_grads = 0
        # weights optimizer

        self.w_optim = torch.optim.Adam(
            self.model.weights(),
            lr=self.config.w_lr,
            betas=(0.0, 0.999),  # config.w_momentum,
            weight_decay=self.config.w_weight_decay,
        )  #

        # architecture optimizer
        self.a_optim = torch.optim.Adam(
            model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )
        self.architect = Architect(
            self.model,
            self.config.w_momentum,
            self.config.w_weight_decay,
            self.config.use_first_order_darts,
        )
    def step(
        self,
        task,
        epoch,
        global_progress="",
        test_phase=False,
        alpha_logger=None,
        sparsify_input_alphas=None,
    ):
        


        log_alphas = False

        if test_phase:
            top1_logger = self.config.top1_logger_test
            losses_logger = self.config.losses_logger_test
            train_steps = self.config.test_task_train_steps
            arch_adap_steps = int(train_steps * self.config.test_adapt_steps)
            
            if alpha_logger is not None:
                log_alphas = True

        else:
            top1_logger = self.config.top1_logger
            losses_logger = self.config.losses_logger
            train_steps = self.config.task_train_steps
            arch_adap_steps = train_steps
            

            

        lr = self.config.w_lr

        if self.config.w_task_anneal:
            for group in self.w_optim.param_groups:
                group["lr"] = self.config.w_lr

            w_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.w_optim, train_steps, eta_min=0.0
            )
        else:
            w_task_lr_scheduler = None

        if self.config.a_task_anneal:
            for group in self.a_optim.param_groups:
                group["lr"] = self.config.alpha_lr

            a_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.a_optim, arch_adap_steps, eta_min=0.0
            )

        else:
            a_task_lr_scheduler = None

        model_has_normalizer = hasattr(self.model, "normalizer")
        if model_has_normalizer:
            self.model.normalizer["params"]["curr_step"] = 0.0
            self.architect.v_net.normalizer["params"]["curr_step"] = 0.0
            self.model.normalizer["params"]["max_steps"] = float(arch_adap_steps)
            self.architect.v_net.normalizer["params"]["max_steps"] = float(
                arch_adap_steps
            )
        from tqdm import tqdm
        if self.config.drop_path_prob > 0.0:
            if not test_phase or self.config.use_drop_path_in_meta_testing:
                self.model.drop_path_prob(self.config.drop_path_prob)

        p_bar = tqdm(range(train_steps))
        self.config.total_steps = train_steps * len(task.train_loader)
        


        for train_step in p_bar:  # task train_steps = epochs per task
            warm_up = (
                epoch < self.warm_up_epochs
            )  # if epoch < warm_up_epochs, do warm up
            if (
                train_step >= arch_adap_steps
            ):  # no architecture adap after arch_adap_steps steps
                warm_up = 1

            if w_task_lr_scheduler is not None:
                w_task_lr_scheduler.step()

            if a_task_lr_scheduler is not None:
                a_task_lr_scheduler.step()
            torch.cuda.reset_peak_memory_stats(device=0)
            
            task_specific_model = train( 
                                                        task,
                                                        self.model,
                                                        self.architect,
                                                        self.w_optim,
                                                        self.a_optim,
                                                        lr,
                                                        global_progress,
                                                        self.config,
                                                        warm_up,
                                                        test_phase
                                                        )
            mem = torch.cuda.memory_stats(0)['allocated_bytes.all.peak']/(1024**2)
            p_bar.set_postfix({"Memory" : f"{mem : .2f}","Task average":f"{self.config.top1_logger_test.avg:.1%}"})
            if train_step == 9:
                self.config.memory_snap = mem
            if (
                model_has_normalizer
                and train_step < (arch_adap_steps - 1)
                and not warm_up
            ): 
                self.model.normalizer["params"]["curr_step"] += 1
                self.architect.v_net.normalizer["params"]["curr_step"] += 1

        w_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_weight)
                for layer_name, layer_weight in self.model.named_weights()
                # if layer_weight.grad is not None
            }
        )
        a_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_alpha)
                for layer_name, layer_alpha in self.model.named_alphas()
                # if layer_alpha.grad is not None
            }
        )

        
        w_task_bot = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_weight)
                for layer_name, layer_weight in task_specific_model.named_weights()
                
            }
        )
        a_task_bot = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_alpha)
                for layer_name, layer_alpha in task_specific_model.named_alphas()
                
        }
        )
        # Log genotype
        genotype = self.model.genotype()

        if log_alphas:
            alpha_logger["normal_relaxed"].append(
                copy.deepcopy(self.model.alpha_normal)
            )
            alpha_logger["reduced_relaxed"].append(
                copy.deepcopy(self.model.alpha_reduce)
            )
            alpha_logger["all_alphas"].append(a_task)
            alpha_logger["normal_hierarchical"].append(
                copy.deepcopy(self.model.alpha_in_normal)
            )
            alpha_logger["reduced_hierarchical"].append(
                copy.deepcopy(self.model.alpha_in_reduce)
            )
            alpha_logger["normal_pairwise"].append(
                copy.deepcopy(self.model.alpha_pw_normal)
            )
            alpha_logger["reduced_pairwise"].append(
                copy.deepcopy(self.model.alpha_pw_reduce)
            )

        # for test data evaluation, turn off drop path
        if self.config.drop_path_prob > 0.0:
            self.model.drop_path_prob(0.0)
        little_switch = 0

        if self.config.naivenaive:
            little_switch = 1
        with torch.no_grad():
            self.config.naivenaive = 1
            self.config.eval_switch = 1
            self.config.cell_phase = 3

            for batch_idx, batch in enumerate(task.test_loader):
                
                x_test, y_test = batch
                x_test = x_test.to(self.config.device, non_blocking=True)
                y_test = y_test.to(self.config.device, non_blocking=True)
                if isinstance(self.model, SearchCNNController):
                    logits = self.model(
                        x_test, sparsify_input_alphas=sparsify_input_alphas
                    )
                else:
                    logits = self.model(x_test)
                loss = self.model.criterion(logits, y_test)

                y_test_pred = logits.softmax(dim=1)
                now =  time.strftime('%c', time.localtime(time.time()))
                prec1, prec5 = utils.accuracy(logits, y_test, self.config, topk=(1, 5))
                losses_logger.update(loss.item(), 1)
                top1_logger.update(prec1.item(), 1)
                
            self.config.naivenaive = 0 
            self.config.eval_switch = 0
            self.config.cell_phase = 3 

            if little_switch == 1:
                self.config.naivenaive = 1
        
        task_info = namedtuple(
            "task_info",
            [
                "genotype",
                "top1",
                "w_task",
                "a_task",
                "loss",
                "y_test_pred",
                "sparse_num_params",
                "w_task_bot",
                "a_task_bot"
            ],
        )
        task_info.w_task = w_task
        task_info.a_task = a_task
        task_info.loss = loss
        y_test_pred = y_test_pred
        task_info.y_test_pred = y_test_pred
        task_info.genotype = genotype
        # task_info.top1 = top1

        # task_info.sparse_num_params = self.model.get_sparse_num_params(
        #     self.model.alpha_prune_threshold
        # )
        task_info.w_task_bot = w_task_bot
        task_info.a_task_bot = a_task_bot

        return task_info

def train(
    task,
    model,
    architect,
    w_optim,
    alpha_optim,
    lr,
    global_progress,
    config,
    warm_up=False,
    test_phase = False
):
    model.train()
    pprevious_grads = list()
    initial_model = copy.deepcopy(model)
    
    p_bar_monitor = (enumerate(zip(task.train_loader, task.valid_loader)))#
    for step, ((train_X, train_y), (val_X, val_y)) in p_bar_monitor:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        train_X, train_y = train_X.to(config.device), train_y.to(config.device)
        val_X, val_y = val_X.to(config.device), val_y.to(config.device)
        N = train_X.size(0)
        initial_alpha = [copy.deepcopy(x).detach().cpu() for x in model.alphas()]
         
        if config.light_exp == 1:

            if config.meta_model != "pc_adaptation" and config.meta_model != "pure_darts" and config.dataset != "cifar10" and config.dataset != "cifar100":
                config.cell_phase = config.layers -1
                architect.v_net.net.config.cell_phase = config.layers -1
            # phase 2. architect step (alpha)
            prohibited_list = config.prohibited_list
            if config.naivenaive != 1 and config.eval_switch != 1 and config.meta_model != "pc_adaptation" and config.meta_model != "pure_darts" and config.dataset not in prohibited_list:

                w_optim.zero_grad()
                alpha_optim.zero_grad()
                train_X, train_y = train_X.chunk(config.split_num), train_y.chunk(config.split_num)
                val_X,val_y = val_X.chunk(config.split_num), val_y.chunk(config.split_num)
                
                for (train_X_chunk, train_y_chunk) ,(val_X_chunk,val_y_chunk) in zip(zip(train_X,train_y),zip(val_X,val_y)):
                    config.cell_phase = config.layers -1
                    architect.v_net.net.config.cell_phase = config.layers -1
                    for phase in range(config.layers):
                
                        if not warm_up:  # only update alphas outside warm up phase
                            if config.do_unrolled_architecture_steps:
                                architect.virtual_step(train_X_chunk, train_y_chunk, lr, w_optim)  # (calc w`)
                            
                            if config.cell_phase == config.layers -1:
                                architect.v_net.net.cells[config.cell_phase].alpha_switch = 1 
                                architect.backward(train_X_chunk, train_y_chunk, val_X_chunk, val_y_chunk, lr, w_optim)
                            
                            
                            else:
                                architect.v_net.net.cells[config.cell_phase].alpha_switch = 1
                                architect.partial_alpha_backward(config, train_X_chunk, train_y_chunk, val_X_chunk, val_y_chunk, lr, w_optim) 
                    
                            
                        model.net.alpha_switch = 0
                        architect.v_net.net.alpha_switch = 0

                        # phase 1. child network step (w)
                        if config.cell_phase == config.layers -1:
                            w_optim.zero_grad()
                            logits = model(train_X_chunk)
                            loss = model.criterion(logits, train_y_chunk)
                            loss_monitor = loss.item()
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)     
                            w_optim.step()


                        else:
                            w_optim.zero_grad()
                            output_grad_sum = copy.deepcopy(config.previous_grad)
                            pprevious_grad = copy.deepcopy(config.pprevious_grad)
                            pprevious_grads.append(pprevious_grad)

                            if config.residual_flag == 1:
                                if config.cell_phase == 1:
                                    if pprevious_grads[0].shape != output_grad_sum.shape:
                                        output_grad_sum = output_grad_sum
                                    else:
                                        output_grad_sum = torch.add(pprevious_grads[0],output_grad_sum)
                                elif config.cell_phase == 0:
                                    if pprevious_grads[1].shape != output_grad_sum.shape:
                                        output_grad_sum = output_grad_sum
                                    else:
                                        output_grad_sum = torch.add(pprevious_grads[1],output_grad_sum)
                            latent = model(train_X_chunk)


                            
                            try:
                                latent.backward(output_grad_sum)
                                
                            except:
                                if output_grad_sum is not None:
                                    print("batch passed,",output_grad_sum.shape, " was the shape of grad saved")
                                    print("what we had to save was this shape, ", latent.shape )
                                    print(f"And this was the phase.{config.cell_phase} what can be the problem here ? ")
                                else:
                                    print("output was none. Why?")
                                pass
                            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
                        

                        
                        config.cell_phase -= 1
                        architect.v_net.net.config.cell_phase -= 1
                    alpha_optim.step() 
                    w_optim.step()
            

            
                
                

        else:
            if not warm_up:  # only update alphas outside warm up phase
                alpha_optim.zero_grad()
                
                if config.do_unrolled_architecture_steps:
                    architect.virtual_step(train_X, train_y, lr, w_optim)  # (calc w`)
                
                architect.backward(train_X, train_y, val_X, val_y, lr, w_optim)
                alpha_optim.step()
                    

                   
            w_optim.zero_grad()
            
            logits = model(train_X)
            
            loss = model.criterion(logits, train_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
            w_optim.step()

            
            


        end.record()
        torch.cuda.synchronize()
        config.computing_time += start.elapsed_time(end)
  
        config.total_steps -= 1
        pprevious_grads = list()
        architect.pprevious_grads = list()
        
        if config.alpha_expect and config.meta_model != 'pc_adaptation':
            if len(config.alpha_grad_footprints) <= 5:

                learnt_alpha = [copy.deepcopy(x).detach().cpu() for x in model.alphas()]
                alpha_grad = _alpha_subtract(initial_alpha,learnt_alpha)
                config.alpha_grad_footprints.append(alpha_grad) 


            else:
                
                learnt_alpha = [copy.deepcopy(x).detach().cpu() for x in model.alphas()]
                alpha_grad = _alpha_subtract(initial_alpha,learnt_alpha)
                
                config.alpha_grad_footprints.pop(0) 
                config.alpha_grad_footprints.append(alpha_grad)   

                config.alpha_sample_metrics = _exp_alpha_metric(initial_alpha,config)
                architect.v_net.net.config.alpha_sample_metrics = config.alpha_sample_metrics

        ###################################################################################


    task_specific_model = copy.deepcopy(model)
    task_specific_model = get_diff_for_const_bottom(initial_model,task_specific_model)
    
    return task_specific_model

def get_diff_for_const_bottom(init_model,task_model):

    get_finite_difference(init_model.named_weights(),task_model.named_weights())

    get_finite_difference(init_model.named_alphas(),task_model.named_alphas())

    return task_model
    

def _alpha_subtract(init,learnt): #Get alpha grad here
    grad_piece = []
    init_copy = init
    learnt_copy = learnt
    for idx in range(len(init)):
        grad_piece.append(torch.subtract(init_copy[idx],learnt_copy[idx]))
        
    return grad_piece
    

def _exp_alpha_metric(initial_alpha,config):  #make metric on theory
    
    alpha_footprint = config.alpha_grad_footprints
    alpha_temp = []
    alpha_mean = []
    for l_idx in range(len(initial_alpha)):
        for s_idx in range(len(alpha_footprint)):
            alpha_temp.append(alpha_footprint[s_idx][l_idx])
        alpha_mean.append(torch.stack(alpha_temp).mean(dim=0))
        alpha_temp = []            
    
    for line in alpha_mean:
        alpha_temp.append(torch.mul(config.total_steps,line))
    
    
    expectation = list()
    for line in range(len(initial_alpha)):
        expectation.append(initial_alpha[line]+alpha_temp[line])
    return expectation
        




def get_meta_arch_power(init_model,task_model):
    # gain alpha and weight difference of this task
    get_finite_difference(init_model.named_weights(),task_model.named_weights())
    get_finite_difference(init_model.named_alphas(),task_model.named_alphas())

    return task_model

    


class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay, use_first_order_darts):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.use_first_order_darts = use_first_order_darts
        self.pprevious_grads = list()
        

    def virtual_step(self, train_X, train_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(train_X, train_y)  # L_train(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        
        



        
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get("momentum_buffer", 0.0) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def backward(self, train_X, train_y, val_X, val_y, xi, w_optim):
        """Compute loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)
        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights, allow_unused=True)
        dalpha = v_grads[: len(v_alphas)]
        dw = v_grads[len(v_alphas) :]

               

        if self.use_first_order_darts:  # use first oder approximation for darts
            
            with torch.no_grad():
                for alpha, da in zip(self.net.alphas(), dalpha):
                    alpha.grad = da
        

        else:  # 2nd order DARTS

            hessian = self.compute_hessian(dw, train_X, train_y)

            # update final gradient = dalpha - xi*hessian
            with torch.no_grad():
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                    alpha.grad = da - xi * h




    def partial_alpha_backward(self,config, train_X, train_y, val_X, val_y, xi, w_optim):
        """Compute loss and backward its gradients
        Args:
           
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # compute gradient
        grad_output_sum = copy.deepcopy(self.v_net.net.config.alpha_previous_grad)
 
        if config.residual_flag == 1:
            pprevious_grad = copy.deepcopy(self.v_net.net.config.alpha_pprevious_grad)
            self.pprevious_grads.append(pprevious_grad) 
            
        latent = self.v_net(val_X)


        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())

        if config.residual_flag == 1:
            try:
                if self.v_net.net.config.cell_phase == 1:
                    grad_output_sum = torch.add(self.pprevious_grads[0],grad_output_sum)

                elif self.v_net.net.config.cell_phase == 0:
                    grad_output_sum = torch.add(self.pprevious_grads[1],grad_output_sum)
            except:
                print(f"Shape error,{grad_output_sum.shape} was the desired shape but you got {self.pprevious_grads[0].shape} or {self.pprevious_grads[1].shape}.")
                print("Bypassing residual flag.")

        v_grads = torch.autograd.grad(latent, v_alphas + v_weights, grad_outputs=grad_output_sum, allow_unused=True) 
        dalpha = v_grads[: len(v_alphas)]
        dw = v_grads[len(v_alphas) :]
        
        

        if self.use_first_order_darts:  # use first oder approximation for darts
            
             with torch.no_grad():
                for alpha, da in zip(self.net.alphas(), dalpha):
                    if alpha.grad is not None and da is not None:
                        alpha.grad.data.add_(da)
                    else:
                        alpha.grad= da

        else:  # 2nd order DARTS

            hessian = self.compute_hessian(dw, train_X, train_y)

            # update final gradient = dalpha - xi*hessian
            with torch.no_grad():
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                    alpha.grad = da - xi * h

    def compute_hessian(self, dw, train_X, train_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_train(w+, alpha) } - dalpha { L_train(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
 
        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        # dalpha { L_train(w+) }
        loss = self.net.loss(train_X, train_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas())

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2.0 * eps * d

        # dalpha { L_train(w-) }
        loss = self.net.loss(train_X, train_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas())

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


