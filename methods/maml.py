from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD
from copy import deepcopy
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
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.reg_const = 1e-4
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=8, num_attention_heads=8,
                            intermediate_size=2048)
        self.meta_model = BertForSequenceClassification(config)
        self.classifier = nn.Linear(args.hidden_size, args.num_labels)
        self.outer_optimizer = SGD(self.meta_model.parameters(), lr=self.outer_update_lr)
        self.meta_model.train()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def forward(self, batch_tasks, training = True, step = 0, epoch=0):
        task_accs = []
        task_loss = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step

        for task_id, task in enumerate(batch_tasks):
            support = task["support"]
            query = task["query"]

            adapter_model = self.classifier
            adapter_model.train()
            self.meta_model.to(self.device)
            adapter_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support), batch_size=self.inner_batch_size)
            inner_optimizer = SGD(adapter_model.parameters(), lr=self.inner_update_lr)
            adapter_model.train()

            all_loss = []
            for inner_step, batch in enumerate(support_dataloader):
                s_batch = {k: v.to(self.device) for k, v in batch.items()}
                embedding_features = self.meta_model(input_ids=s_batch["input_ids"],
                                                     attention_mask=s_batch["attention_mask"])
                outputs = adapter_model(embedding_features)
                loss = self.criterion(outputs.view(-1, self.num_labels), s_batch["labels"]) + self.reg_const * sum([x.norm().pow(2) for x in adapter_model.parameters()])
                if training:
                    loss.backward(retain_graph=True, create_graph=True)
                else:
                    loss.backward()
                inner_optimizer.step()
                all_loss.append(loss.item())

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).__next__()
            q_batch = {k: v.to(self.device) for k, v in query_batch.items()}
            q_embedding_features = self.meta_model(input_ids=q_batch["input_ids"],
                                                   attention_mask=q_batch["attention_mask"])
            q_outputs = adapter_model(q_embedding_features)
            q_loss = self.criterion(q_outputs.view(-1, self.num_labels), q_batch["labels"])

            if training:
                q_loss.backward()
                for i, params in enumerate(self.meta_model.parameters()):
                    if task_id == 0:
                        sum_gradients.append(deepcopy(params.grad.detach()))
                    else:
                        sum_gradients[i] += deepcopy(params.grad.detach())
            self.outer_optimizer.zero_grad()
            inner_optimizer.zero_grad()
            q_logits = F.softmax(q_outputs,dim=1)
            pre_label_id = torch.argmax(q_logits,dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_batch["labels"].detach().cpu().numpy().tolist()
            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            task_loss.append(q_loss.detach().cpu())

            del adapter_model, inner_optimizer
            torch.cuda.empty_cache()
            print(f'{self.args.methods} Task loss: {np.mean(task_loss):.4f}')

        if training:
            # Average gradient across tasks
            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.meta_model.parameters()):
                params.grad = sum_gradients[i]
                if params.grad is None:
                    continue
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            del sum_gradients
            gc.collect()

        return np.mean(task_accs), np.mean(task_loss)

