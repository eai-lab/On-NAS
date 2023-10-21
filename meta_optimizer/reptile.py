""" Script for On-NAS & Two-Fold Meta-learning(TFML) & On-NAS

This code have been written for a research purpose. 

Licenses and code references will be added at camera-ready version of the code. 

"""


from collections import OrderedDict,defaultdict
import torch
from utils import utils
import copy
import torch.nn as nn
class NAS_Reptile:
    
    def __init__(self, meta_model,meta_cell, config):

        self.meta_model = meta_model
        self.config = config
        self.meta_cell = meta_cell
        
        if config.w_meta_optim is None:
            self.w_meta_optim = torch.optim.Adam(
                self.meta_model.weights(), lr=self.config.w_meta_lr
            )

        else:
            self.w_meta_optim = self.config.w_meta_optim

        assert meta_cell is not None ,"meta_cell is empty, not being passed, abort."
            

        if config.a_meta_optim is None:
            if meta_model.alphas() is not None:
                print("found alphas, set meta optim")
                self.a_meta_optim = torch.optim.Adam(
                    self.meta_model.alphas(), lr=self.config.a_meta_lr
                )
            else:
                print("-------- no alphas, no meta optim ------")

        else:
            self.a_meta_optim = self.config.a_meta_optim

        # additional optimizer for the cell itself. let it change as outer lr changes 
        self.w_meta_cell_optim = torch.optim.SGD(meta_cell.parameters(), lr = config.w_meta_lr)


        
    def step(self, task_infos, meta_cell):

        # Extract infos provided by the task_optimizer
        # k_way = task_infos[0].k_way
        # data_shape = task_infos[0].data_shape

        w_tasks = [task_info.w_task for task_info in task_infos]
        a_tasks = [task_info.a_task for task_info in task_infos]

        w_tasks_bot = [task_info.w_task_bot for task_info in task_infos]  
        a_tasks_bot = [task_info.a_task_bot for task_info in task_infos] 
        
        
        # create denominator for coefficients. 
        w_const_bottom = get_coeff_bottom(w_tasks_bot)
        a_const_bottom = get_coeff_bottom(a_tasks_bot)

        self.w_meta_optim.zero_grad()
        self.a_meta_optim.zero_grad()

        self.meta_model.train()
        
 
        w_finite_differences = list()
        a_finite_differences = list()

        w_const_list = list()
        a_const_list = list()

        w_cell_finite_difference = list()
        
        for task_info in task_infos:
            w_finite_differences += [
                get_finite_difference(self.meta_model.named_weights(), task_info.w_task)
            ]
            a_finite_differences += [
                get_finite_difference(self.meta_model.named_alphas(), task_info.a_task)
            ]
        

        if self.config.exp_const:

            w_finite_const_applied = list()
            a_finite_const_applied = list()

            for idx, task_info in enumerate(task_infos):
                w_const_list += [
                    get_const(w_finite_differences[idx],w_const_bottom)
                    ]
                a_const_list += [
                    get_const(a_finite_differences[idx],a_const_bottom)
                ]
            
            
            w_finite_const_applied = get_const_applied_list(w_finite_differences,
                                                            w_const_list,
                                                            self.config.layers,
                                                            self.config.const_mult)
            a_finite_const_applied = get_const_applied_list(a_finite_differences,
                                                            a_const_list,
                                                            self.config.layers,
                                                            self.config.const_mult)

            


            mean_w_task_finitediff_with_const = {
                k: get_mean_gradient_from_key(k, w_finite_const_applied)
                for idx, k in enumerate(w_tasks[0].keys())
            }

            mean_a_task_finitediff_with_const = {
                k: get_mean_gradient_from_key(k, a_finite_const_applied)
                for idx, k in enumerate(a_tasks[0].keys())
            }
            
            
            for layer_name, layer_weight_tensor in self.meta_model.named_weights():
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-(mean_w_task_finitediff_with_const[layer_name])) #*w_const_list[task_idx][layer_name])
                

            for layer_name, layer_weight_tensor in self.meta_model.named_alphas():
                
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-(mean_a_task_finitediff_with_const[layer_name]))
                

            self.w_meta_optim.step()
            if self.a_meta_optim is not None:
                self.a_meta_optim.step()
                    
        
        else:
            
            mean_w_task_finitediff = {
                k: get_mean_gradient_from_key(k, w_finite_differences)
                for k in w_tasks[0].keys()
            }

            mean_a_task_finitediff = {
                k: get_mean_gradient_from_key(k, a_finite_differences)
                for k in a_tasks[0].keys()
            }
            for layer_name, layer_weight_tensor in self.meta_model.named_weights():
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-mean_w_task_finitediff[layer_name])
                
            
            
            for layer_name, layer_weight_tensor in self.meta_model.named_alphas():
                if layer_weight_tensor.grad is not None:
                    layer_weight_tensor.grad.data.add_(-mean_a_task_finitediff[layer_name])
                

            self.w_meta_optim.step()
            if self.a_meta_optim is not None:
                self.a_meta_optim.step()


            
        if self.config.exp_cell:
            
            cell_const_container = make_cell_coeff(self.meta_model, w_const_bottom)
            
            for idx in range(len(self.meta_model.net.cells)):           #calculate difference
                meta_model_cell_dict = make_dict(self.meta_model.net.cells[idx].named_parameters())
                w_cell_finite_difference += [
                    get_finite_difference(meta_cell.named_parameters(), meta_model_cell_dict)
                ]
            
            w_cell_finite_const_applied = get_const_applied_list(w_cell_finite_difference,cell_const_container,self.config.layers,self.config.cell_const_mult)

            mean_w_cell_dict = {
                k:get_mean_gradient_from_key(k, w_cell_finite_const_applied)
                for idx,k in enumerate(meta_model_cell_dict.keys())
            }
    
            

            mean_w_cell_wo_const = {
                k:get_mean_gradient_from_key(k,w_cell_finite_difference)
                for idx,k in enumerate(meta_model_cell_dict.keys())
            }

            if self.config.cell_const_flag:
                target_dict = mean_w_cell_dict
            else:
                target_dict = mean_w_cell_wo_const


            for idx in range(len(self.meta_model.net.cells)):           
                for layer_name, layer_weight_tensor in self.meta_cell.named_parameters():
                    if layer_weight_tensor.grad is not None:
                        layer_weight_tensor.grad.data.add_(-target_dict[layer_name])

            self.w_meta_cell_optim.step()
            alter_weights(self.meta_model, self.meta_cell)





def make_dict(generator_entity):

    dictified = {
        layer_name: copy.deepcopy(layer_weight)
        for layer_name, layer_weight in generator_entity
    }
    return dictified

                


def get_coeff_bottom(dict_collections):
    '''so result is basically model-shaped dictionary which have sqrt of exp sum for weight differences.'''
    result = defaultdict(float)

    for task_bot in dict_collections:
        for layer_name,weight_values in task_bot.items():
            result[layer_name] += torch.exp(weight_values)

    for layer_name,weight_values in result.items():
        result[layer_name] = weight_values


    return result





def get_const(task_specific_info,const_bottom):
    result = defaultdict(float)
    
    for layer_name,_ in task_specific_info.items():
        result[layer_name] = torch.div(torch.exp(task_specific_info[layer_name]),
                                       torch.sum(const_bottom[layer_name],dim=0))
        
    return result

def alter_weights(meta_model, meta_cell):
    '''
    Altering meta-model weights into the meta-cell weight, regardless of location.
    
    '''
    
    mmw = meta_model.net.cells
    mcw = list(meta_cell.named_weights())
    change_count =0

    for cell in mmw:                                                        # for each cell,
        for layer_name, layer_weight_tensor in cell.named_weights():        # iterate over all keys
            if layer_weight_tensor is not None:
                for meta_layer_name, meta_layer_weight_tensor in mcw:             
                    if meta_layer_name == layer_name:
                        layer_weight_tensor.data = meta_layer_weight_tensor 
                        change_count +=1
        
# some helper functions


def get_finite_difference(meta_weights, task_weights):
    for layer_name, layer_weight_tensor in meta_weights:
        if layer_weight_tensor.grad is not None:
            task_weights[layer_name].data.sub_(layer_weight_tensor.data)
        
    return task_weights  # = task weights - meta weights


def get_mean_gradient_from_key(k, task_gradients):
    grad_with_keys = []
    for grad in task_gradients:
        try:
            grad_with_keys.append(grad[k])
        except:
            pass
    try:
        stacked_mean = torch.stack(grad_with_keys).mean(dim=0)
    except:
        temp_grads = []
        for cell in grad_with_keys:
            if isinstance(cell,float):
                continue
            else:
                temp_grads.append(cell)
        stacked_mean = torch.stack(temp_grads).mean(dim=0)
    
    return stacked_mean



def apply_const(finite_diff_at_task,const_dict_at_task,const_mult=1):
    const_result = defaultdict(float)

    for layer_name, layer_weight_tensor in finite_diff_at_task.items():
        const_result[layer_name] = layer_weight_tensor * const_dict_at_task[layer_name] * const_mult
    
    return const_result

        
def get_const_applied_list(diffs,consts,n_layer,const_mult=1):
    result = list()

    if isinstance(consts,list):
        for idx, task_diff in enumerate(diffs):
            result += [apply_const(task_diff,consts[idx],const_mult)]

    else:
        cell_const_list = list()
        for idx in range(n_layer):
            const_dict = defaultdict(float)
            for layer_name,layer_weight_tensor in consts.net.cells[idx].named_parameters():
                const_dict[layer_name] = layer_weight_tensor
            cell_const_list.append(const_dict)
        for idx, task_diff in enumerate(diffs):
            result += [apply_const(task_diff,cell_const_list[idx],const_mult)]
    return result



def make_cell_coeff(model, const_bottom): 
    
    model_container = copy.deepcopy(model) #make model-shaped container
    
    for layer_name, layer_weight_tensor in model_container.named_weights(): #fill value with const bottom
        layer_weight_tensor = const_bottom[layer_name]

    im_cell = defaultdict(float)
    for layer_name, layer_weight_tensor in model_container.net.cells[0].named_weights():
        im_cell[layer_name] = []

     # make contatiner for calculation
    for idx in range(len(model.net.cells)): 
        cell_weight_gen = model_container.net.cells[idx].named_weights() 
        for layer_name, layer_weight_tensor in cell_weight_gen: 
            im_cell[layer_name] += torch.exp(layer_weight_tensor) #
            
    # now we got the sum ( vertical )
    for idx in range(len(model.net.cells)):
        numerator = model_container.net.cells[idx].named_weights()
        for layer_name, layer_weight_tensor in numerator:
            layer_weight_tensor = torch.div(torch.exp(layer_weight_tensor),im_cell[layer_name][0])
            
    return model_container





    
        
        
        

    