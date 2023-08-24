# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import pandas as pd;
import numpy as np;
import torch
from transformers import AdamW, get_scheduler
from collections import defaultdict

import engine
from model import RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC, DeBertaFGBC, GPT_Neo13FGBC
from dataset import DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo, DatasetDeberta, DatasetGPT_Neo13
from evaluate import test_evaluate

import utils
from visualize import save_acc_loss_curves   #save_acc_curves, save_loss_curves 
from dataset import train_validate_test_split

import utils
import matplotlib.pyplot as plt
from Model_Config import Model_Config, traits
import os

os.chdir('/home/bruce/dev/dissertation-binary/Scripts')


def run(args: Model_Config):
    print("This is the model name ", args.pretrained_model)
    print("type that args is in run method ", type(args))
    print("This is the args.dataset_path in train run method", args.dataset_path)
    
    # Read in the new per trait data sets
    # All traits will be compared to the same class '0' Notcb - Not Cyberbullying
    # All dataframes will have two output classes

    # LOOP needed, running through with one trait first hardcoded
    #print(os.getcwd())
    # Get the absolute path to the file
    #file_path = os.path.abspath("../Dataset/Binary/train/train_Age.csv")
    os.chdir('/home/bruce/dev/dissertation-binary/Scripts')
    #print(pd.read_csv(file_path).head())
    

    # the loop would get all the single traits 
    all_traits = {} 
    for i in range(0,6):
        all_traits[''.join(['train_',str(i)])] = pd.read_csv(''.join([args.dataset_path, 'train/train_', traits.get(str(i)), '.csv'   ])).dropna()
        all_traits[''.join(['val_',str(i)])] = pd.read_csv(''.join([args.dataset_path, 'val/val_', traits.get(str(i)), '.csv'   ])).dropna()
        all_traits[''.join(['test_',str(i)])] = pd.read_csv(''.join([args.dataset_path, 'test/test_', traits.get(str(i)), '.csv'   ])).dropna()

    #print(all_traits)

    cb_trts = [0, 1, 2, 4, 5]
    for trt in cb_trts:
        #trt = 1     # Hand checking all the traits
        train_df =  all_traits.get(''.join(['train_',str(trt)]))
        valid_df =  all_traits.get(''.join(['val_',str(trt)]))
        test_df =   all_traits.get(''.join(['test_',str(trt)])) 

        # Set the target values of the trait to 0
        train_df['target'].replace(trt,0, inplace=True)
        valid_df['target'].replace(trt,0, inplace=True)
        test_df['target'].replace(trt,0, inplace=True)

        # Append single trait with Notcb 3
        # As of pandas 2.0, append (previously deprecated) was removed.
        # You need to use concat instead (for most applications):
        trt_ncb = 3
        loc = [''.join(['train_',str(trt_ncb)]),
               ''.join(['val_',str(trt_ncb)]),
               ''.join(['test_',str(trt_ncb)])]

        #train_df = pd.concat([train_df, pd.DataFrame(all_traits.get(loc[0]))],ignore_index = True)
        #valid_df = pd.concat([train_df, pd.DataFrame(all_traits.get(loc[1]))],ignore_index = True)
        #train_df = pd.concat([train_df, pd.DataFrame(all_traits.get(loc[2]))],ignore_index = True)
        train_df = train_df._append(all_traits.get(''.join(['train_',str(trt_ncb)])),ignore_index = True)
        valid_df = valid_df._append(all_traits.get(''.join(['val_',str(trt_ncb)])),ignore_index = True)
        test_df = test_df._append(all_traits.get(''.join(['test_',str(trt_ncb)])),ignore_index = True)
        print("length of test_df is ", len(test_df))

        # Set the target values to 1 for the Notcb
        
        train_df['target'].replace(trt_ncb,1, inplace=True)
        valid_df['target'].replace(trt_ncb,1, inplace=True)
        test_df['target'].replace(trt_ncb,1, inplace=True)
        
        #print(train_df)
        
        #print(set(train_df.label.values))
        #print("train len - {}, valid len - {}, test len - {}".format(len(train_df),\
        #len(valid_df),len(test_df)))
        #for col in train_df.columns:
            #print(col)
        #print("train example text -- ",train_df.text[1],"\nwith target -- ",\
        #train_df.label[1])
        
        # BHG Text encoding occurs at model instantiation  
        train_dataset = generate_dataset(train_df, args)
        #print(train_dataset.target)
        #print("train_dataset object is of type -- ",type(train_dataset))
        #print("Print Encoded Token Byte tensor at location 1 -- ", train_dataset[1]['input_ids'])

        #print(train_df['target'].nunique())
        
        #encoding = train_dataset[1]['input_ids']
        # TODO - the line below does not print out in Windows python due to unicode error would need
        #        to convert each of the items to ascii or utf-8
        # print("The Decoded Token Text tensor is -- ",train_dataset.tokenizer.convert_ids_to_tokens(encoding))
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = args.train_batch_size,
            shuffle = True
        )

        valid_dataset = generate_dataset(valid_df, args)
        valid_data_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset,
            batch_size = args.valid_batch_size,
            shuffle = True
        )

        test_dataset = generate_dataset(test_df, args)
        test_data_loader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = args.test_batch_size,
            shuffle = False
        )
        
        device = utils.set_device(args)

        model = set_model(args)
        # BHG model type and number of parameters initial instantiation
        print("Model Class: ", type(model), "Num Params: ",count_model_parameters(model))

        model = model.to(device)
        
        # BHG Model Paramter definition
        num_train_steps = int(len(train_df) / args.train_batch_size * args.epochs)

        #print(model.named_parameters())
        
        # BHG definition of named_parameters from PyTorch documentation
        # Returns an iterator over module parameters, yielding both the name 
        #   of the parameter as well as the parameter itself.

        # Weight_Decay: Manually setting the weight_decay of the model paramters based on the
        # name of the parameter

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # As per the Kaggle On Stability of a Few-Samples tutorial you should not 
        # Also blanket override the weight_decay if it is declared conditionaly in
        # the optimizer_parameter dictionary
        #https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning
        #
        # optimizer = AdamW(
        #     optimizer_grouped_parameters,
        #     lr=lr,
        #     eps=epsilon,
        #     correct_bias=not use_bertadam # bias correction step - not needed default is True
        # )
        # However in running a test the results came out exactly the same so it must be
        # smart enough to know to use the weights correctly as defined in the optimizer_parameters
        #
        
        optimizer = AdamW(
            params = optimizer_parameters,
            lr = args.learning_rate,
            weight_decay = args.weight_decay,
            eps = args.adamw_epsilon
        )

        scheduler = get_scheduler(
            "linear",
            optimizer = optimizer,
            num_warmup_steps = num_train_steps*0.2,
            num_training_steps = num_train_steps
        )

        print("---Starting Training---")
        
        history = defaultdict(list)
        best_acc = 0.0
        
        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')
            print('-'*10)

            train_acc, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler, args)
            print(f'Epoch {epoch + 1} --- Training loss: {train_loss} Training accuracy: {train_acc}')
            val_acc, val_loss = engine.eval_fn(valid_data_loader, model, device, args)
            print(f'Epoch {epoch + 1} --- Validation loss: {val_loss} Validation accuracy: {val_acc}')
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            # SAVE MODEL if best so far going through epochs
            if val_acc>best_acc:
                print(f'Epoch {epoch + 1} val_acc {val_acc} best_acc {best_acc} trait {traits.get(str(trt))}')
                torch.save(model.state_dict(), f"{args.model_path}{traits.get(str(trt))}_Best_Val_Acc.bin")
                # BHG needed to set best_acc to val_acc this was missing in prior implementation
                best_acc=val_acc

            
        print(f'\n---History---\n{history}')
        print("##################################### Testing ############################################")
        pred_test, acc = test_evaluate(trt,test_df, test_data_loader, model, device, args)
        pred_test.to_csv(f'{args.output_path}{traits.get(str(trt))}---test_acc---{acc}.csv', index = False)

        # Create and save the Accuracy and Loss plot during epoch training per trait
        plt_acc_loss = save_acc_loss_curves(args, trt, history)
        
        del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("##################################### Task End ############################################")    
    
def create_dataset_files(args):
    if args.dataset == "FGBC":
        df = pd.read_csv(f'{args.dataset_path}dataset.csv').dropna()

        if args.classes == 5:
            indexnames = df[ df['label'] == 'Notcb' ].index
            df = df.drop(indexnames , inplace=False)
            df = df.reset_index()
            df.loc[df['target']==5, "target"] = 3
        print("This is the length of df in FIVE ",len(df))
    elif args.dataset == "Twitter":
        df = pd.read_csv(f'{args.dataset_path}twitter_dataset.csv').dropna()

    #Splitting the dataset
    train_df, valid_df, test_df = train_validate_test_split(df)
    train_df.to_csv(f'{args.dataset_path}train.csv')
    valid_df.to_csv(f'{args.dataset_path}valid.csv')
    test_df.to_csv(f'{args.dataset_path}test.csv')


def generate_dataset(df, cur_args: Model_Config):
    if(cur_args.pretrained_model == "microsoft/deberta-v3-base"):
        return DatasetDeberta(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "EleutherAI/gpt-neo-125m"):
        return DatasetGPT_Neo(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "EleutherAI/gpt-neo-1.3B"):
        return DatasetGPT_Neo13(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model== "roberta-base"):
        return DatasetRoberta(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model== "xlnet-base-cased"):
        return DatasetXLNet(text=df.text.values, target=df.target.values, args=cur_args)
    elif(cur_args.pretrained_model == "albert-base-v2"):
        return DatasetAlbert(text=df.text.values, target=df.target.values, args=cur_args)
    

def set_model(args):
    # BHG debug
    print("The model in the args is ", args.pretrained_model)
    
    if(args.pretrained_model == "microsoft/deberta-v3-base"):
        return DeBertaFGBC(args)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-125m"):
        return GPT_NeoFGBC(args)
    elif(args.pretrained_model == "EleutherAI/gpt-neo-1.3B"):
        return GPT_Neo13FGBC(args)
    elif(args.pretrained_model == "roberta-base"):
        return RobertaFGBC(args)
    elif(args.pretrained_model == "xlnet-base-cased"):
        return XLNetFGBC(args)
    elif(args.pretrained_model == "albert-base-v2"):
        return AlbertFGBC(args)

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

