

DATASET=cifar10
DATASET_DIR=../dataset
TRAIN_DIR=./found_architecture_checkpoint
		
mkdir -p $TRAIN_DIR




    args=(
        # Execution
        --name metatest_in \
        --seed 21 #good :21 35,51,86 
        --path ${TRAIN_DIR} \
        --data_path ${DATASET_DIR} \
        --dataset $DATASET
        --hp_setting 'in_metanas' \
        --use_hp_setting 1 \
        --workers 0 \
        --gpus 0 \
        --test_adapt_steps 1.0 \
        --batch_size 64 \

        # number classes  
        --k 10 \

        # Meta Learning
        --meta_model searchcnn\
        --epochs 50 \
        --warm_up_epochs  25\
        --use_pairwise_input_alphas \
        --eval_freq 0 \
        --eval_epochs 100 \
        --normalizer softmax \
        --normalizer_temp_anneal_mode linear \
        --normalizer_t_min 0.1 \
        --normalizer_t_max 1.0 \
        --drop_path_prob 0.2 \
        --test_task_train_steps 100

        # Architectures
        --init_channels 28 \
        --layers 4 \
        --reduction_layers 1 3 \
        --use_first_order_darts \

        # experiments
        --exp_const 1
        --exp_cell 1
        --wandb 0
        --wandb_name d
        --const_mult 10
        --unittest 0
        --cell_const_flag 1
        --cell_const_mult 4
        --light_exp 1
        --opsample 1
        --sampleno 2
        --naivenaive 0
        --cell_phase 3
        --residual_flag 0
        --beta_sampling 0
        --eval_switch 0
        --alpha_expect 1
        --split_num 5
        --w_lr 0.025
        --exp_name CIFAR
        
    )

    python3 cifar_search.py "${args[@]}" 


