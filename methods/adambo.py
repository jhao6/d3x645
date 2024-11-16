from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD, Adam
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from bert_model import BertForSequenceClassification
from transformers import BertConfig
class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        self.data=args.data
        self.args.neumann_lr = 1e-4
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.reg_const = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048)

        self.meta_model = BertForSequenceClassification(config)
        self.adapter_model = nn.Linear(args.hidden_size, args.num_labels)

        param_count = 0
        for param in self.adapter_model.parameters():
            param_count += param.numel()
        # initialize the outer and inner optimizer
        self.outer_optimizer = Adam(self.meta_model.parameters(), lr=self.outer_update_lr, betas=(0.9, 0.999), weight_decay=0.01, eps=args.lambd)
        # self.inner_optimizer = Adam(self.adapter_model.parameters(), lr=self.outer_update_lr, betas=(0.9, 0.999), eps=1e-8)
        self.inner_optimizer = SGD(self.adapter_model.parameters(), lr=self.inner_update_lr)
        self.meta_model.train()
        self.adapter_model.train()
        # The number of steps for lower-level initialization
        self.y_warm_start = args.y_warm_start
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training=True, step=0, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.y_warm_start if (epoch==0 or not training) else self.inner_update_step

        for taskid, task in enumerate(batch_tasks):
            support = task["support"]
            query = task["query"]
            self.meta_model.to(self.device)
            self.adapter_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            all_loss = []
            for _ in range(num_inner_update_step):
                for batch in support_dataloader:
                    s_batch = {k: v.to(self.device) for k, v in batch.items()}
                    embedding_features = self.meta_model(input_ids=s_batch["input_ids"], attention_mask=s_batch["attention_mask"])
                    outputs = self.adapter_model(embedding_features)
                    inner_loss = self.criterion(outputs.view(-1, self.num_labels), s_batch["labels"]) + self.reg_const * sum([x.norm().pow(2) for x in self.adapter_model.parameters()])
                    inner_loss.backward()
                    self.inner_optimizer.step()
                    self.inner_optimizer.zero_grad()
                    print(f'inner loss: {inner_loss:.4f}')
            all_loss.append(inner_loss)
            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).__next__()
            q_batch = {k: v.to(self.device) for k, v in query_batch.items()}
            q_embedding_features = self.meta_model(input_ids=q_batch["input_ids"], attention_mask=q_batch["attention_mask"])
            q_outputs = self.adapter_model(q_embedding_features)
            q_loss = self.criterion(q_outputs.view(-1, self.num_labels), q_batch["labels"])
            if training:
                hypergrad = hypergradient(self.args, q_loss, self.meta_model, self.adapter_model, q_batch, s_batch, self.reg_const)
                # print(f'Task loss: {np.mean(all_loss):.4f}')
                for i, params in enumerate(hypergrad):
                    if taskid == 0:
                        sum_gradients.append(params.detach())
                    else:
                        sum_gradients[i] += params.detach()
            q_logits = F.softmax(q_outputs, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_batch["labels"].detach().cpu().numpy().tolist()
            self.outer_optimizer.zero_grad()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())
            torch.cuda.empty_cache()
            print(f'{self.args.methods} Task loss: {np.mean(task_loss):.4f}')

        if training:
            # Average gradient across tasks
            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            # Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.meta_model.parameters()):
                params.grad = sum_gradients[i]
                if params.grad is None:
                    continue
            # update the upper-level variables
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            del sum_gradients
            gc.collect()

        return np.mean(task_accs),  np.mean(task_loss)



def hypergradient(args, out_loss, meta_model, adapter_model, query_batch, support_batch, reg_const):
    Fy_gradient = torch.autograd.grad(out_loss, adapter_model.parameters(), retain_graph=True)
    F_gradient = [g_param.view(-1) for g_param in Fy_gradient]
    v_0 = torch.unsqueeze(torch.reshape(torch.hstack(F_gradient), [-1]), 1).detach()
    Fx_gradient = torch.autograd.grad(out_loss, meta_model.parameters())

    # calculate the neumann series
    z_list = []
    s_embedding_features = meta_model(input_ids=support_batch["input_ids"], attention_mask=support_batch["attention_mask"])
    outputs = adapter_model(s_embedding_features)
    inner_loss = F.cross_entropy(outputs.view(-1, args.num_labels), support_batch["labels"]) + reg_const * sum(
        [x.norm().pow(2) for x in adapter_model.parameters()])
    G_gradient = []
    Gy_gradient = torch.autograd.grad(inner_loss, adapter_model.parameters(), create_graph=True)
    for g_grad, param in zip(Gy_gradient, adapter_model.parameters()):
        G_gradient.append((param - args.neumann_lr * g_grad).view(-1))
    G_gradient = torch.reshape(torch.hstack(G_gradient), [-1])

    for _ in range(args.hessian_q):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, adapter_model.parameters(), retain_graph=True)
        v_params = [v_param.view(-1) for v_param in v_new]
        v_0 = torch.unsqueeze(torch.reshape(torch.hstack(v_params), [-1]), 1).detach()
        z_list.append(v_0)
    v_Q = args.neumann_lr * (v_0 + torch.sum(torch.stack(z_list), dim=0))

    # Gyx_gradient
    q_embedding_features = meta_model(input_ids=query_batch["input_ids"], attention_mask=query_batch["attention_mask"])
    q_outputs = adapter_model(q_embedding_features)
    q_loss = F.cross_entropy(q_outputs.view(-1, args.num_labels), query_batch["labels"]) + reg_const * sum(
        [x.norm().pow(2) for x in adapter_model.parameters()])
    Gy_gradient = torch.autograd.grad(q_loss, adapter_model.parameters(), retain_graph=True, create_graph=True)
    Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
    Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
    Gyxv_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, v_Q.detach()), meta_model.parameters())
    outer_update = [x + y for (x, y) in zip(Fx_gradient, Gyxv_gradient)]

    return outer_update









