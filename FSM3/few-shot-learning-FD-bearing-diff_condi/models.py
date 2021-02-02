import os, sys
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from backbone import FeatureExtractor_2, FeatureExtractor_4, Flatten

class Baseline_FinetuneLast(nn.Module):
    # basic CNN classify model, with two classifiers in it. Used for source task and target task with different num of types
    # we finetune only the classifier in this model

    def __init__(self, args):
        super(Baseline_FinetuneLast, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        self.backbone = FeatureExtractor_4(args=args)
        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        self.linear_source = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_source_num_classes)
        self.linear_target = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_target_num_classes)

        self.pretrain_classify_model = nn.Sequential(self.backbone, self.linear_source)

        self.backbone.to(self.gpu_device)
        self.pretrain_classify_model.to(self.gpu_device)  # exploit defined model to required GPU.
        self.linear_source.to(self.gpu_device)
        self.linear_target.to(self.gpu_device)

        # loss_function
        self.classify_loss = nn.CrossEntropyLoss()

        # optimizer and lr_scheduler
        # self.optimizer = optim.Adam(self.model_stack.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)
        self.optimizer = optim.Adam(self.pretrain_classify_model.parameters(), lr=0.001)
        self.optimizer_finetune = optim.Adam(self.linear_target.parameters(), lr=0.0001)

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.8)

    def train_classify_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            self.optimizer.zero_grad()

            y_pred_prob = self.pretrain_classify_model(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss.backward()

            self.optimizer.step()

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))
        return np.mean(loss_list), np.mean(accu_list)


    def test_classify_iter(self, data_loader, epoch, type='Val'):

        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            y_pred_prob = self.model_stack(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        test_loss, test_accu = np.mean(loss_list), np.mean(accu_list)

        print('Epoch {:d} | {} Loss {:f} | {} Accu {:f}'.format(epoch, type, test_loss, type, test_accu))

    def test_iter_meta_finetune(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):

            self.linear_target.reset_parameters()

            x = x.unsqueeze(2).float()
            y = y.long()
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.to(self.gpu_device)
            y = y.to(self.gpu_device)

            n_way = x.size(0)

            x_support = x[:, :self.args.n_shot, :, :]  # [n_way, n_shot, 1, 2048]
            x_query = x[:, self.args.n_shot:, :]  # [n_way, n_query, 1, 2048]

            x_support = x_support.contiguous().view(n_way * self.args.n_shot, *x_support.size()[2:])  # [n_way*n_shot, 1, 2048]
            x_query = x_query.contiguous().view(n_way * self.args.n_query, *x_query.size()[2:])  # [n_way*n_query, 1, 2048]

            y_support = y[:, :self.args.n_shot]
            y_query = y[:, self.args.n_shot:]

            y_support = y_support.contiguous().view(n_way * self.args.n_shot, 1).squeeze(1)  # [n_way*n_shot]
            y_query = y_query.contiguous().view(n_way * self.args.n_query, 1).squeeze(1)  # [n_way*n_query]


            # descide whether to finetune whole model or only classifier
            self.backbone.train(False)
            self.linear_target.train(True)

            # finetune model with support data from new domain
            for n_f in range(self.args.pretrain_finetune_steps):
                x_support_feat = self.backbone(x_support)
                x_support_feat = x_support_feat.detach()
                y_support_pred_prob = self.linear_target(x_support_feat)
                support_loss = self.classify_loss(y_support_pred_prob, y_support)
                self.optimizer_finetune.zero_grad()
                support_loss.backward()
                self.optimizer_finetune.step()

            # evaluate
            self.linear_target.train(False)
            x_query_feat = self.backbone(x_query)
            y_query_pred_prob = self.linear_target(x_query_feat)

            loss = self.classify_loss(y_query_pred_prob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_prob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        # record
        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | Loss {:f} | Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, np.mean(loss_list), acc_mean, confi_interval))

        return acc_mean, confi_interval


class Baseline_FinetuneWhole(nn.Module):
    # basic CNN classify model, with two classifiers in it. Used for source task and target task with different num of types
    # we finetune the whole model in this version
    def __init__(self, args):
        super(Baseline_FinetuneWhole, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        self.backbone = FeatureExtractor_4(args=args)
        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        self.linear_source = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_source_num_classes)
        self.linear_target = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_target_num_classes)

        self.pretrain_classify_model = nn.Sequential(self.backbone, self.linear_source)

        self.target_classify_model = nn.Sequential(self.backbone, self.linear_target)

        self.pretrain_classify_model.to(self.gpu_device)  # exploit defined model to required GPU.
        self.target_classify_model.to(self.gpu_device)  # exploit defined model to required GPU.
        self.linear_source.to(self.gpu_device)
        self.linear_target.to(self.gpu_device)

        # loss_function
        self.classify_loss = nn.CrossEntropyLoss()

        # optimizer and lr_scheduler
        self.optimizer = optim.Adam(self.pretrain_classify_model.parameters(), lr=0.001)
        self.optimizer_finetune = optim.Adam(self.target_classify_model.parameters(), lr=0.0001)

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.8)


    def train_classify_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            self.optimizer.zero_grad()

            y_pred_prob = self.pretrain_classify_model(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss.backward()

            self.optimizer.step()

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))
        return np.mean(loss_list), np.mean(accu_list)


    def test_classify_iter(self, data_loader, epoch, type='Val'):

        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            y_pred_prob = self.model_stack_source(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        test_loss, test_accu = np.mean(loss_list), np.mean(accu_list)

        print('Epoch {:d} | {} Loss {:f} | {} Accu {:f}'.format(epoch, type, test_loss, type, test_accu))

    def test_iter_meta_finetune(self, data_loader, epoch, result_dir):
        loss_list, accu_list = [], []

        backup_para = {}
        # save current model parameter
        # for name, para in self.backbone.named_parameters():
        #     backup_para[name] = para.data.copy_()
        torch.save(self.backbone.state_dict(), os.path.join(result_dir, 'backbone_backup.pth'))

        for i, (x, y) in enumerate(data_loader):

            self.linear_target.reset_parameters()

            x = x.unsqueeze(2).float()
            y = y.long()
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.to(self.gpu_device)
            y = y.to(self.gpu_device)

            n_way = x.size(0)

            x_support = x[:, :self.args.n_shot, :, :]  # [n_way, n_shot, 1, 2048]
            x_query = x[:, self.args.n_shot:, :]  # [n_way, n_query, 1, 2048]

            x_support = x_support.contiguous().view(n_way * self.args.n_shot, *x_support.size()[2:])  # [n_way*n_shot, 1, 2048]
            x_query = x_query.contiguous().view(n_way * self.args.n_query, *x_query.size()[2:])  # [n_way*n_query, 1, 2048]

            y_support = y[:, :self.args.n_shot]
            y_query = y[:, self.args.n_shot:]

            y_support = y_support.contiguous().view(n_way * self.args.n_shot, 1).squeeze(1)  # [n_way*n_shot]
            y_query = y_query.contiguous().view(n_way * self.args.n_query, 1).squeeze(1)  # [n_way*n_query]

            # descide whether to finetune whole model or only classifier
            self.target_classify_model.train(True)

            # finetune model with support data from new domain
            for n_f in range(self.args.pretrain_finetune_steps):
                y_support_pred_prob = self.target_classify_model(x_support)
                support_loss = self.classify_loss(y_support_pred_prob, y_support)
                self.optimizer_finetune.zero_grad()
                support_loss.backward()
                self.optimizer_finetune.step()

            # evaluate
            self.target_classify_model.train(False)
            y_query_pred_prob = self.target_classify_model(x_query)

            loss = self.classify_loss(y_query_pred_prob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_prob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

            # restore model and continue pretrain
            self.backbone.load_state_dict(torch.load(os.path.join(result_dir, 'backbone_backup.pth')))

        # record
        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | Loss {:f} | Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, np.mean(loss_list), acc_mean, confi_interval))

        return acc_mean, confi_interval


class Baseline_FeatureKnn(nn.Module):
    # basic CNN classify model, with two classifiers in it. Used for source task and target task with different num of types
    # we only use the trained feature extractor and classify the query data by matching to support data (KNN classifier)
    def __init__(self, args):
        super(Baseline_FeatureKnn, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        self.backbone = FeatureExtractor_4(args=args)
        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        self.linear_source = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_source_num_classes)

        self.pretrain_classify_model = nn.Sequential(self.backbone, self.linear_source)


        self.pretrain_classify_model.to(self.gpu_device)  # exploit defined model to required GPU.
        self.linear_source.to(self.gpu_device)

        # loss_function
        self.classify_loss = nn.CrossEntropyLoss()

        # optimizer and lr_scheduler
        self.optimizer = optim.Adam(self.pretrain_classify_model.parameters(), lr=0.001)

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.8)


    def train_classify_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            self.optimizer.zero_grad()

            y_pred_prob = self.pretrain_classify_model(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss.backward()

            self.optimizer.step()

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))
        return np.mean(loss_list), np.mean(accu_list)


    def test_classify_iter(self, data_loader, epoch, type='Val'):

        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            y_pred_prob = self.model_stack(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        test_loss, test_accu = np.mean(loss_list), np.mean(accu_list)

        print('Epoch {:d} | {} Loss {:f} | {} Accu {:f}'.format(epoch, type, test_loss, type, test_accu))

    def test_iter_meta_finetune(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):

            x = x.unsqueeze(2).float()
            y = y.long()
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.to(self.gpu_device)
            y = y.to(self.gpu_device)

            n_way = x.size(0)

            x_support = x[:, :self.args.n_shot, :, :]  # [n_way, n_shot, 1, 2048]
            x_query = x[:, self.args.n_shot:, :]  # [n_way, n_query, 1, 2048]

            x_support = x_support.contiguous().view(n_way * self.args.n_shot, *x_support.size()[2:])  # [n_way*n_shot, 1, 2048]
            x_query = x_query.contiguous().view(n_way * self.args.n_query, *x_query.size()[2:])  # [n_way*n_query, 1, 2048]

            y_support = y[:, :self.args.n_shot]
            y_query = y[:, self.args.n_shot:]

            y_support = y_support.contiguous().view(n_way * self.args.n_shot, 1)  # [n_way*n_shot]
            y_query = y_query.contiguous().view(n_way * self.args.n_query, 1).squeeze(1)  # [n_way*n_query]

            # descide whether to finetune whole model or only classifier
            self.backbone.train(False)

            # classify by matching query to support data

            y_query_pred_prob = self.knn_matching(x_support, x_query, y_support, n_way)

            loss = self.classify_loss(y_query_pred_prob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_prob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        # record
        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | Loss {:f} | Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, np.mean(loss_list), acc_mean, confi_interval))

        return acc_mean, confi_interval

    def knn_matching(self, x_support, x_query, y_support, n_way):
        x_feat_support = self.backbone(x_support)
        x_feat_query = self.backbone(x_query)

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim =1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim =1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm+ 0.00001)

        # get support label in one-hot form
        y_support_onehot = torch.zeros((y_support.size()[0], n_way)).scatter_(1, y_support.cpu(), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob =(weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob

class MatchingNetwork(nn.Module):
    # perform episodic matching network training in original data space
    def __init__(self, args):
        super(MatchingNetwork, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        self.backbone = FeatureExtractor_4(args=args)

        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        self.backbone.to(self.gpu_device) # exploit defined model to required GPU.

        self.param_list = list(self.backbone.parameters())

        # loss_function
        self.metric_loss = nn.NLLLoss()

        # optimizer and lr_scheduler
        self.optimizer = optim.Adam(self.param_list)


    def forward_fce(self, x):
        n_way = x.size(0)

        x = x.contiguous().view(n_way * (self.args.n_shot + self.args.n_query), *x.size()[2:])  # [n_way*(n_shot+n_query), 1, 2048]

        x_feat = self.backbone(x)  # [n_way*(n_shot+n_query), f_dim]

        x_feat = x_feat.view(n_way, (self.args.n_shot + self.args.n_query), -1)  # [n_way, n_shot+n_query, f_dim]

        x_feat_support = x_feat[:, :self.args.n_shot, :]  # [n_way, n_shot, f_dim]
        x_feat_query = x_feat[:, self.args.n_shot:, :]  # [n_way, n_query, f_dim]

        x_feat_support = x_feat_support.contiguous().view(n_way*self.args.n_shot, -1)  # [n_way*n_shot, f_dim]
        x_feat_query = x_feat_query.contiguous().view(n_way*self.args.n_query, -1)  # [n_way*n_query, dim]

        # encoder support feature with bi-direction LSTM
        support_feat_enc = self.support_feat_encoder(x_feat_support.unsqueeze(0))[0]
        support_feat_enc = support_feat_enc.squeeze(0)  # [n_way*n_shot, 2*f_dim] 2 means bi-direction
        x_feat_support = x_feat_support + support_feat_enc[:,:x_feat.size(-1)] + support_feat_enc[:,x_feat.size(-1):]

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim =1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # encoder query feature with FCE
        x_feat_query = self.FCE(x_feat_query, x_feat_support)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim =1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm+ 0.00001)

        # get support label in one-hot form
        y_support = torch.from_numpy(np.repeat(range(n_way), self.args.n_shot))
        y_support_onehot = torch.zeros((n_way*self.args.n_shot, n_way)).scatter_(1, y_support.unsqueeze(1), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob =(weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob


    def forward_no_fce(self, x):
        n_way = x.size(0)

        x = x.contiguous().view(n_way * (self.args.n_shot + self.args.n_query), *x.size()[2:])  # [n_way*(n_shot+n_query), 1, 2048]

        x_feat = self.backbone(x)  # [n_way*(n_shot+n_query), f_dim]

        x_feat = x_feat.view(n_way, (self.args.n_shot + self.args.n_query), -1)  # [n_way, n_shot+n_query, f_dim]

        x_feat_support = x_feat[:, :self.args.n_shot, :]  # [n_way, n_shot, f_dim]
        x_feat_query = x_feat[:, self.args.n_shot:, :]  # [n_way, n_query, f_dim]

        x_feat_support = x_feat_support.contiguous().view(n_way*self.args.n_shot, -1)  # [n_way*n_shot, f_dim]
        x_feat_query = x_feat_query.contiguous().view(n_way*self.args.n_query, -1)  # [n_way*n_query, dim]

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim =1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim =1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm+ 0.00001)

        # get support label in one-hot form
        y_support = torch.from_numpy(np.repeat(range(n_way), self.args.n_shot))
        y_support_onehot = torch.zeros((n_way*self.args.n_shot, n_way)).scatter_(1, y_support.unsqueeze(1), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob =(weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob


    def train_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.unsqueeze(2).float()
            n_way = x.size(0)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.forward_no_fce(x)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            loss_list.append(loss.data.cpu().numpy())

            y_query_pred = np.argmax(y_query_pred_logprob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_query_pred == y_query.data.cpu().numpy(), dtype=np.float32) / y_query_pred.shape[0]

            accu_list.append(batch_accu)

        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))

    def test_iter(self, data_loader, epoch, type):
        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            x = x.unsqueeze(2).float()
            # x is one few-shot learning task with data shape: [5, 1+16, 3, 224, 224]
            n_way = x.size(0)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.forward_no_fce(x)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_logprob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)


            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | {} Loss {:f} | {} Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, type, np.mean(loss_list), type, acc_mean, confi_interval))

        return acc_mean, confi_interval

class MatchingNetwork_feat_space(nn.Module):
    # perform episodic matching network training in pretrained feature space
    def __init__(self, args):
        super(MatchingNetwork_feat_space, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        #### define backbone
        self.backbone = FeatureExtractor_2(args=args)

        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        # self.backbone.to(self.gpu_device) # exploit defined model to required GPU.

        #### define pretrain module
        flatten_pre = Flatten()
        fc_pre = nn.Linear(64*(self.args.data_size//256), self.args.pretrain_source_num_classes)

        self.pretrain_classify_model = nn.Sequential(self.backbone, flatten_pre, fc_pre)

        #### define matching network module
        conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        relu1 = nn.ReLU()

        conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        relu2 = nn.ReLU()

        flatten_metric = Flatten()
        fc_metric = nn.Linear(64 * (self.args.data_size // 256), self.args.backbone_out_dim)

        self.metric_backbone = nn.Sequential(conv1, relu1, conv2, relu2, flatten_metric, fc_metric)

        self.metric_model = nn.Sequential(self.backbone, self.metric_backbone)

        #### put all modules to GPU
        self.backbone.to(self.gpu_device)
        self.pretrain_classify_model.to(self.gpu_device)
        self.metric_backbone.to(self.gpu_device)
        self.metric_model.to(self.gpu_device)

        self.param_backbone = list(self.backbone.parameters())
        self.param_classify_model = list(self.pretrain_classify_model.parameters())
        self.param_metric_backbone = list(self.metric_backbone.parameters())

        # loss_function
        self.classify_loss = nn.CrossEntropyLoss()
        self.metric_loss = nn.NLLLoss()

        # optimizer and lr_scheduler

        self.optimizer_classify_model = optim.Adam(self.param_classify_model)
        self.optimizer_metric_model = optim.Adam(self.param_metric_backbone)


    def metric_forward(self, x):
        n_way = x.size(0)

        x = x.contiguous().view(n_way * (self.args.n_shot + self.args.n_query), *x.size()[2:])  # [n_way*(n_shot+n_query), 1, 2048]

        x_feat_basic = self.backbone(x)  # [n_way*(n_shot+n_query), 64, 8]

        x_feat_basic = x_feat_basic.detach()  # we keep the backbone fixed

        x_feat = self.metric_backbone(x_feat_basic)  # [n_way*(n_shot+n_query), f_dim]

        x_feat = x_feat.view(n_way, (self.args.n_shot + self.args.n_query), -1)  # [n_way, n_shot+n_query, f_dim]

        x_feat_support = x_feat[:, :self.args.n_shot, :]  # [n_way, n_shot, f_dim]
        x_feat_query = x_feat[:, self.args.n_shot:, :]  # [n_way, n_query, f_dim]

        x_feat_support = x_feat_support.contiguous().view(n_way*self.args.n_shot, -1)  # [n_way*n_shot, f_dim]
        x_feat_query = x_feat_query.contiguous().view(n_way*self.args.n_query, -1)  # [n_way*n_query, dim]

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim=1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim=1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm + 0.00001)

        # get support label in one-hot form
        y_support = torch.from_numpy(np.repeat(range(n_way), self.args.n_shot))
        y_support_onehot = torch.zeros((n_way*self.args.n_shot, n_way)).scatter_(1, y_support.unsqueeze(1), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob = (weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob

    def backbone_knn_forward(self, x):
        n_way = x.size(0)

        x = x.contiguous().view(n_way * (self.args.n_shot + self.args.n_query), *x.size()[2:])  # [n_way*(n_shot+n_query), 1, 2048]

        x_feat = self.backbone(x)  # [n_way*(n_shot+n_query), 64, 8]

        x_feat = x_feat.view(n_way * (self.args.n_shot + self.args.n_query), -1)

        x_feat = x_feat.view(n_way, (self.args.n_shot + self.args.n_query), -1)  # [n_way, n_shot+n_query, f_dim]

        x_feat_support = x_feat[:, :self.args.n_shot, :]  # [n_way, n_shot, f_dim]
        x_feat_query = x_feat[:, self.args.n_shot:, :]  # [n_way, n_query, f_dim]

        x_feat_support = x_feat_support.contiguous().view(n_way*self.args.n_shot, -1)  # [n_way*n_shot, f_dim]
        x_feat_query = x_feat_query.contiguous().view(n_way*self.args.n_query, -1)  # [n_way*n_query, dim]

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim=1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim=1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm + 0.00001)

        # get support label in one-hot form
        y_support = torch.from_numpy(np.repeat(range(n_way), self.args.n_shot))
        y_support_onehot = torch.zeros((n_way*self.args.n_shot, n_way)).scatter_(1, y_support.unsqueeze(1), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob = (weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob




    def train_classify_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            self.optimizer_classify_model.zero_grad()

            y_pred_prob = self.pretrain_classify_model(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss.backward()

            self.optimizer_classify_model.step()

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))

        return np.mean(loss_list), np.mean(accu_list)


    def train_metric_iter(self, data_loader, epoch):
        self.backbone.train(False)
        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.unsqueeze(2).float()
            n_way = x.size(0)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.metric_forward(x)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            self.optimizer_metric_model.zero_grad()

            loss.backward()

            self.optimizer_metric_model.step()

            loss_list.append(loss.data.cpu().numpy())

            y_query_pred = np.argmax(y_query_pred_logprob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_query_pred == y_query.data.cpu().numpy(), dtype=np.float32) / y_query_pred.shape[0]

            accu_list.append(batch_accu)

        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))

    def test_metric_iter(self, data_loader, epoch, type):
        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            x = x.unsqueeze(2).float()
            # x is one few-shot learning task with data shape: [5, 1+16, 3, 224, 224]
            n_way = x.size(0)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.metric_forward(x)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_logprob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | {} Loss {:f} | {} Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, type, np.mean(loss_list), type, acc_mean, confi_interval))

        return acc_mean, confi_interval

    def test_backbone_knn_iter(self, data_loader, epoch, type):
        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            x = x.unsqueeze(2).float()
            # x is one few-shot learning task with data shape: [5, 1+16, 3, 224, 224]
            n_way = x.size(0)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.backbone_knn_forward(x)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_logprob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96* acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | {} Loss {:f} | {} Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, type, np.mean(loss_list), type, acc_mean, confi_interval))

        return acc_mean, confi_interval

    def save_feature(self, data_loader, feat_save_path, save_num=-1, type='Train'):
        # save_num: how many feature points of each class do we save, if -1 means save all, type train or test
        if type not in ['Train', 'Val', 'Test']:
            raise ValueError('Wrong type. Must be Train or Val or Test.')

        feat_save = {}

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()  # [b, 1, l]
            x = x.to(self.gpu_device)  # put data to GPU.

            y = y.squeeze(1).data.numpy()  # [b]
            n = x.size()[0]

            feat = self.metric_model(x)  # [b, feat_dim]
            feat = feat.data.cpu().numpy()

            for k in range(n):
                l = y[k]

                if l not in feat_save:
                    feat_save[l] = []
                feat_save[l].append(feat[k])

        for k, v in feat_save.items():
            v = v[0:save_num]
            feat_save[k] = np.array(v)

        f = open(os.path.join(feat_save_path, '{}_{}_{}shot_{}_feat.pkl'.format(self.args.source_dataset, self.args.target_dataset, self.args.n_shot, type)), 'wb')
        pickle.dump(feat_save, f)
        f.close()

class MatchingNetwork_pretrain(nn.Module):
    # pretrain backbone as initial of matching network
    def __init__(self, args):
        super(MatchingNetwork_pretrain, self).__init__()
        self.args = args

        self.gpu_device = get_avaliable_gpu(self.args)
        print('Using GPU: {}.'.format(self.gpu_device))

        #### define backbone
        self.backbone = FeatureExtractor_4(args=args)

        self.backbone_pretrain_path = args.backbone_pretrain_path
        if self.backbone_pretrain_path != '':
            # need to load pretrained parameters for backbone
            self.backbone.load_state_dict(torch.load(self.backbone_pretrain_path))
        else:
            # randomly initialize
            self.backbone.apply(model_weights_init)

        #### define pretrain module
        fc_pre = nn.Linear(self.args.backbone_out_dim, self.args.pretrain_source_num_classes)

        self.pretrain_classify_model = nn.Sequential(self.backbone, fc_pre)

        #### put all modules to GPU
        self.backbone.to(self.gpu_device)
        self.pretrain_classify_model.to(self.gpu_device)

        self.param_backbone = list(self.backbone.parameters())
        self.param_classify_model = list(self.pretrain_classify_model.parameters())

        # loss_function
        self.classify_loss = nn.CrossEntropyLoss()
        self.metric_loss = nn.NLLLoss()

        # optimizer and lr_scheduler

        self.optimizer_classify_model = optim.Adam(self.param_classify_model)
        self.optimizer_metric_model = optim.Adam(self.param_backbone, 0.0001)

    def metric_forward(self, x):
        n_way = x.size(0)

        x = x.contiguous().view(n_way * (self.args.n_shot + self.args.n_query), *x.size()[2:])  # [n_way*(n_shot+n_query), 1, 2048]

        x_feat = self.backbone(x)  # [n_way*(n_shot+n_query), f_dim]

        x_feat = x_feat.view(n_way, (self.args.n_shot + self.args.n_query), -1)  # [n_way, n_shot+n_query, f_dim]

        x_feat_support = x_feat[:, :self.args.n_shot, :]  # [n_way, n_shot, f_dim]
        x_feat_query = x_feat[:, self.args.n_shot:, :]  # [n_way, n_query, f_dim]

        x_feat_support = x_feat_support.contiguous().view(n_way*self.args.n_shot, -1)  # [n_way*n_shot, f_dim]
        x_feat_query = x_feat_query.contiguous().view(n_way*self.args.n_query, -1)  # [n_way*n_query, dim]

        # get normalized support feature
        x_feat_support_norm = torch.norm(x_feat_support, p=2, dim=1).unsqueeze(1).expand_as(x_feat_support)
        x_feat_support_normalized = x_feat_support.div(x_feat_support_norm+ 0.00001)

        # get normalized query feature
        x_feat_query_norm = torch.norm(x_feat_query, p=2, dim=1).unsqueeze(1).expand_as(x_feat_query)
        x_feat_query_normalized = x_feat_query.div(x_feat_query_norm + 0.00001)

        # get support label in one-hot form
        y_support = torch.from_numpy(np.repeat(range(n_way), self.args.n_shot))
        y_support_onehot = torch.zeros((n_way*self.args.n_shot, n_way)).scatter_(1, y_support.unsqueeze(1), 1)
        y_support_onehot = y_support_onehot.to(self.gpu_device)  # [n_way*n_shot, n_way]

        # calculate similarity between support and query sample
        similar_mat = nn.ReLU()(x_feat_query_normalized.mm(x_feat_support_normalized.transpose(0,1)))*100  # follow setting of Closer Look at FSL
        weight_mat = nn.Softmax()(similar_mat)

        # predict query sample label
        y_query_logprob = (weight_mat.mm(y_support_onehot)+1e-6).log()

        return y_query_logprob


    def train_classify_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, y) in enumerate(data_loader):
            x = x.unsqueeze(1).float()
            y = y.squeeze(1).long()
            x, y = x.to(self.gpu_device), y.to(self.gpu_device)  # put data to GPU.

            self.optimizer_classify_model.zero_grad()

            y_pred_prob = self.pretrain_classify_model(x)

            loss = self.classify_loss(y_pred_prob, y)

            loss.backward()

            self.optimizer_classify_model.step()

            loss_list.append(loss.data.cpu().numpy())

            y_pred = np.argmax(y_pred_prob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_pred == y.data.cpu().numpy(), dtype=np.float32) / y_pred.shape[0]

            accu_list.append(batch_accu)

        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))

        return np.mean(loss_list), np.mean(accu_list)


    def train_metric_iter(self, data_loader, epoch):
        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            # x is one few-shot learning task with data shape: [n_way, n_shot+n_query, 1, 2048]
            x = x.unsqueeze(2).float()
            n_way = x.size(0)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.metric_forward(x)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            self.optimizer_metric_model.zero_grad()

            loss.backward()

            self.optimizer_metric_model.step()

            loss_list.append(loss.data.cpu().numpy())

            y_query_pred = np.argmax(y_query_pred_logprob.data.cpu().numpy(), axis=1)

            batch_accu = np.sum(y_query_pred == y_query.data.cpu().numpy(), dtype=np.float32) / y_query_pred.shape[0]

            accu_list.append(batch_accu)

        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {:d} | Ave Loss {:f} | Ave Train Accu {:f}'.format(epoch, np.mean(loss_list), np.mean(accu_list)))

    def test_metric_iter(self, data_loader, epoch, type):
        if type not in ['Val', 'Test']:
            raise ValueError('Wrong test type. Must be Val or Test.')

        loss_list, accu_list = [], []

        for i, (x, _) in enumerate(data_loader):
            x = x.unsqueeze(2).float()
            # x is one few-shot learning task with data shape: [5, 1+16, 3, 224, 224]
            n_way = x.size(0)

            y_query = torch.from_numpy(np.repeat(range(n_way), self.args.n_query))  # y_query: n_way*n_query vector
            y_query = y_query.to(self.gpu_device)

            x = x.to(self.gpu_device)  # put data to GPU.

            y_query_pred_logprob = self.metric_forward(x)

            loss = self.metric_loss(y_query_pred_logprob, y_query)

            loss_list.append(loss.data.cpu().numpy())

            # we get top-k results
            topk_scores, topk_labels = y_query_pred_logprob.data.topk(k=1, dim=1, largest=True, sorted=True)
            topk_pred = topk_labels.squeeze(1).cpu().numpy()
            num_correct = np.sum(topk_pred == y_query.data.cpu().numpy()).astype(np.float32)
            num_all = y_query.size(0)

            batch_accu = (num_correct / num_all) * 100

            accu_list.append(batch_accu)

        acc_all  = np.asarray(accu_list)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        confi_interval = 1.96 * acc_std/np.sqrt(i+1)
        print('Epoch {:d} | Task num: {} | {} Loss {:f} | {} Accu {:.3f}+-{:.3f}.'.format(epoch, i+1, type, np.mean(loss_list), type, acc_mean, confi_interval))

        return acc_mean, confi_interval

class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim*2, feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = torch.zeros(1,feat_dim)
        self.feat_dim = feat_dim
        #self.K = K

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0)
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h

    def to(self, device):
        super(FullyContextualEmbedding, self).to(device)
        self.c_0 = self.c_0.to(device)


def model_weights_init(m):
    classname = m.__class__.__name__
    # print ('model layer : %s' % classname)

    if classname.find('Conv1d') != -1:
        # init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')

        if m.bias is not None:
            init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

def get_avaliable_gpu(args):
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        raise ValueError('No available gpu. Check your system.')

    if use_gpu and args.gpu_id:
        # print('Using GPU %s.' % args.gpu_id)
        gpu_str = 'cuda:' + args.gpu_id
        gpu_device = torch.device(gpu_str)
        return gpu_device

