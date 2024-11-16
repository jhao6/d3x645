from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
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
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.reg_const = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=8, num_attention_heads=8,
                            intermediate_size=2048)
        self.meta_model = BertForSequenceClassification(config)
        self.adapter_model = nn.Linear(args.hidden_size, args.num_labels)
        param_count = 0
        for param in self.adapter_model.parameters():
            param_count += param.numel()
        self.z_params = torch.randn(param_count, 1)
        self.z_params = nn.init.xavier_uniform_(self.z_params).to(self.device)
        self.outer_optimizer = SGD(self.meta_model.parameters(), lr=self.outer_update_lr)
        self.inner_optimizer = SGD(self.adapter_model.parameters(), lr=self.inner_update_lr)
        self.meta_model.train()
        self.adapter_model.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training=True, step=0, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_task = len(batch_tasks)

        for taskid, task in enumerate(batch_tasks):
            support = task["support"]
            query = task["query"]
            self.meta_model.to(self.device)
            self.adapter_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support), batch_size=self.inner_batch_size)

            all_loss = []
            inner_loss = torch.zeros(1).to(self.device)
            for idx, batch in enumerate(support_dataloader):
                s_batch = {k: v.to(self.device) for k, v in batch.items()}
                embedding_features = self.meta_model(input_ids=s_batch["input_ids"],
                                                     attention_mask=s_batch["attention_mask"])
                outputs = self.adapter_model(embedding_features)
                inner_loss += self.criterion(outputs.view(-1, self.num_labels), s_batch["labels"]) + self.reg_const * sum(
                    [x.norm().pow(2) for x in self.adapter_model.parameters()])
            inner_loss.backward(retain_graph=True, create_graph=True)
            all_loss.append(inner_loss.item())
            g_grad = [g_param.grad.view(-1) for g_param in self.adapter_model.parameters()]
            g_grad_flat = torch.unsqueeze(torch.reshape(torch.hstack(g_grad), [-1]), 1)

            jvp = torch.autograd.grad(g_grad_flat, self.adapter_model.parameters(), grad_outputs=self.z_params)
            self.inner_optimizer.step()
            self.inner_optimizer.zero_grad()
            jvp = [j_param.detach().view(-1) for j_param in jvp]
            jvp_flat = torch.unsqueeze(torch.reshape(torch.hstack(jvp), [-1]), 1)

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).__next__()
            q_batch = {k: v.to(self.device) for k, v in query_batch.items()}
            q_embedding_features = self.meta_model(input_ids=q_batch["input_ids"], attention_mask=q_batch["attention_mask"])
            q_outputs = self.adapter_model(q_embedding_features)
            q_loss = self.criterion(q_outputs.view(-1, self.num_labels), q_batch["labels"])
            if training:
                hypergrad, self.z_params = hypergradient(self.args, jvp_flat, self.z_params, \
                                                                   q_loss, self.meta_model, self.adapter_model, s_batch, self.reg_const)
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

            # update meta parameters:
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            del sum_gradients
            gc.collect()

        return np.mean(task_accs),  np.mean(task_loss)


def hypergradient(args, jacob_flat, z_params, loss, meta_model, adapter_model, support_batch, reg_const):
    loss.backward()
    Fy_gradient = [g_param.grad.detach().view(-1) for g_param in adapter_model.parameters()]
    Fx_gradient = [g_param.grad.detach() for g_param in meta_model.parameters()]
    Fy_gradient_flat = torch.unsqueeze(torch.reshape(torch.hstack(Fy_gradient), [-1]), 1)
    z_params -= args.nu * (jacob_flat - Fy_gradient_flat)
    # Gyx_gradient
    s_embedding_features = meta_model(input_ids=support_batch["input_ids"],
                                      attention_mask=support_batch["attention_mask"])
    outputs = adapter_model(s_embedding_features)
    inner_loss = F.cross_entropy(outputs.view(-1, args.num_labels), support_batch["labels"]) + reg_const * sum(
        [x.norm().pow(2) for x in adapter_model.parameters()])
    Gy_gradient = torch.autograd.grad(inner_loss, adapter_model.parameters(), retain_graph=True, create_graph=True)
    Gy_params = [Gy_param.view(-1) for Gy_param in Gy_gradient]
    Gy_gradient_flat = torch.reshape(torch.hstack(Gy_params), [-1])
    Gyxz_gradient = torch.autograd.grad(-torch.matmul(Gy_gradient_flat, z_params.detach()), meta_model.parameters())
    hyper_grad = [Fx + Gyxz for (Fx, Gyxz) in zip(Fx_gradient, Gyxz_gradient)]

    return hyper_grad, z_params

