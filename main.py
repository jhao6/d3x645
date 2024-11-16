import torch
from transformers import BertTokenizer, Trainer, TrainingArguments

from datasets import load_dataset
import numpy as np
import os
import argparse
import time
import math
from construct_task import construct_meta_task
from methods import adambo, stocbio, maml, anil, ttsa, saba, ma_soba, bo_rep, slip
import random
def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle=True, batch_size=20):
    idxs = list(range(0, len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size, len(taskset)))]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='trec', type=str,
                        help="dataset for meta-learning", )

    parser.add_argument("--save_direct", default='trec', type=str,
                        help="Path to save file")

    parser.add_argument("--methods", default='adambo', type=str,
                        help="choise method [maml, anil, stocbio, ttsa, saba, ma_soba, bo_rep, slip, adambo]")

    parser.add_argument("--num_labels", default=6, type=int,
                        help="Number of class for classification")

    parser.add_argument("--hidden_size", default=768, type=int,
                        help="hidden size of bert")

    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of outer interation")

    parser.add_argument("--k_spt", default=25, type=int,
                        help="Number of support samples per task")

    parser.add_argument("--outer_batch_size", default=20, type=int,
                        help="Batch of task size")

    parser.add_argument("--inner_batch_size", default=50, type=int,
                        help="Training batch size in inner iteration")

    parser.add_argument("--outer_update_lr", default=1e-3, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=1e-3, type=float,
                        help="Inner update learning rate")

    parser.add_argument("--neumann_lr", default=1e-3, type=float,
                        help="update for hessian")

    parser.add_argument("--hessian_q", default=5, type=int,
                        help="Q steps for hessian-inverse-vector product")

    parser.add_argument("--inner_update_step", default=1, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--num_task_train", default=500, type=int,
                        help="Total number of meta tasks for training")

    parser.add_argument("--num_task_test", default=200, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--grad_clip", default=False, type=bool,
                        help="whether grad clipping or not")

    parser.add_argument("--grad_normalized", default=True, type=bool,
                        help="whether grad normalized or not")

    parser.add_argument("--gamma", default=1e-3, type=float,
                        help="clipping threshold")

    parser.add_argument("--lambd", default=1e-8, type=float,
                        help="eps for adam")

    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")

    parser.add_argument("--beta", default=0.90, type=float,
                        help="momentum parameters")

    parser.add_argument("--nu", default=1e-2, type=float,
                        help="learning rate of z")

    parser.add_argument("--y_warm_start", default=3, type=int,
                        help="update steps for y")

    parser.add_argument("--interval", default=2, type=int,
                        help="update interval for y")

    args = parser.parse_args()
    random_seed(args.seed)
    st = time.time()

    if args.methods == 'maml':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-3
        learner = maml.Learner(args)

    if args.methods == 'anil':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 2e-2
        learner = anil.Learner(args)

    if args.methods == 'stocbio':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 2e-3
        args.hessian_q = 3
        args.neumann_lr = 1e-3
        args.inner_update_step = 3
        args.inner_batch_size = 100
        learner = stocbio.Learner(args)
    #
    if args.methods == 'ttsa':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-3
        args.neumann_lr = 1e-3
        learner = ttsa.Learner(args)
    #
    if args.methods == 'saba':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-2
        args.nu = 1e-2
        learner = saba.Learner(args)

    elif args.methods == 'ma_soba':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-2
        args.beta = 0.9
        args.nu = 1e-2
        learner = ma_soba.Learner(args)

    if args.methods == 'bo_rep':
        args.outer_update_lr = 1e-1
        args.inner_update_lr = 5e-2
        args.grad_normalized = True
        args.y_warm_start = 3
        args.interval = 2
        args.beta = 0.9
        args.nu = 1e-2
        args.inner_update_step = 1
        learner = bo_rep.Learner(args)
    #
    if args.methods == 'slip':
        args.outer_update_lr = 1e-1
        args.inner_update_lr = 5e-2
        args.y_warm_start = 3
        args.grad_normalized = True
        args.grad_clip = False
        args.interval = 1
        args.beta = 0.9
        args.nu = 1e-2
        args.inner_update_step = 1
        learner = slip.Learner(args)

    if args.methods == 'adambo':
        args.inner_update_lr =1e-4
        args.outer_update_lr = 5e-3
        args.inner_update_step = 1
        args.y_warm_start = 3
        args.hessian_q = 3
        learner = adambo.Learner(args)

    # load data
    data = load_dataset("trec", trust_remote_code=True)

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # data preprocessing function
    def preprocess_data(examples):
        encodings = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)
        encodings["labels"] = examples["coarse_label"]
        return encodings

    encoded_dataset = data.map(preprocess_data)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_task = construct_meta_task(encoded_dataset["train"])
    test_task = construct_meta_task(encoded_dataset["test"])

    acc_test = []
    loss_test = []
    acc_train = []
    loss_train = []
    print(args)
    for epoch in range(args.epochs):
        print(f'Meta learning method: {args.methods}')
        print(f"[epoch/epochs]:{epoch}/{args.epochs}")
        db = create_batch_of_tasks(train_task, is_shuffle=True, batch_size=args.outer_batch_size)
        acc_task = []
        loss_task = []
        for step, task_batch in enumerate(db):
            print(f'\n---------Epoch: {epoch}, Step {step} -----------')
            acc, loss = learner(task_batch, training=True, step=step, epoch=epoch)
            acc_task.append(acc)
            loss_task.append(loss)
        acc_train.append(float(format(np.mean(acc), '.4f')))
        loss_train.append(float(format(np.mean(loss), '.4f')))
        print(f'{args.methods}, Epoch: {epoch}, training Loss: {loss_train}')
        print(f'{args.methods}, Epoch: {epoch}, training Acc: {acc_train}')

        print("---------- Testing Mode -------------")
        db_test = create_batch_of_tasks(test_task, is_shuffle=False, batch_size=100)
        acc_task = []
        loss_task = []
        for test_batch in db_test:
            acc, loss = learner(test_batch, training=False, epoch=epoch)
            acc_task.append(acc)
            loss_task.append(loss)
        acc_test.append(float(format(np.mean(acc_task), '.4f')))
        loss_test.append(float(format(np.mean(loss_task), '.4f')))
        print(f'{args.methods}, Epoch: {epoch}, Test Loss: {loss_test}')
        print(f'{args.methods}, Epoch: {epoch}, Test Acc: {acc_test}')

    file_name = f'{args.methods}_outlr{args.outer_update_lr}_inlr{args.inner_update_lr}_beta{args.beta}_gamma{args.gamma}_lambda{args.lambd}_seed{args.seed}'
    args.save_direct = os.path.join(args.save_direct, args.methods)
    args.save_direct = os.path.join(args.save_direct, 'lambda')
    if not os.path.exists('logs/' + args.save_direct):
        os.mkdir('logs/' + args.save_direct)
    save_path = 'logs/' + args.save_direct
    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name) + '.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Train ACC': str(acc_train),
                     'AVG Test ACC': str(acc_test), 'AVG Train LOSS': str(loss_train), 'AVG Test LOSS': str(loss_test),
                     'time': total_time}))
    files.close()
    torch.save((acc_train, acc_test, loss_train, loss_test), os.path.join(save_path, file_name))
    print(args)
    print(f'time:{total_time} h')


if __name__ == "__main__":
    main()
