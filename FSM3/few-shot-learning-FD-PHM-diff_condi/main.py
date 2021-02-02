import os, sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn

from Dataload import SimpleDataLoader, MetaDataLoader
from models import Baseline_FinetuneLast, Baseline_FinetuneWhole, Baseline_FeatureKnn, Baseline_FeatureKnn_Proto
from models import MatchingNetwork, MatchingNetwork_feat_space, MatchingNetwork_pretrain
from models import ProtoNetwork, ProtoNetwork_feat_space, ProtoNetwork_pretrain
from options import parse_args

from report import Tap, create_result_subdir, export_sources


if __name__ == "__main__":

    ######### load settings ##########
    args = parse_args()

    ################## for training record ######################
    stdout_tap = Tap(sys.stdout)
    stderr_tap = Tap(sys.stderr)
    sys.stdout = stdout_tap
    sys.stderr = stderr_tap

    result_subdir = create_result_subdir(os.path.join(args.base_logdir, args.method, 'TRAIN'), 'exp')
    print("Saving logs to {}".format(result_subdir))

    # Start dumping stdout and stderr into result directory.
    stdout_tap.set_file(open(os.path.join(result_subdir, 'stdout.txt'), 'wt'))
    stderr_tap.set_file(open(os.path.join(result_subdir, 'stderr.txt'), 'wt'))

    # Saving source files.
    export_sources(os.path.join(result_subdir, 'src'))

    # Saving model parameters.
    model_save_path = os.path.join(result_subdir, 'saved_model')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # Saving data feature.
    feat_save_path = os.path.join(args.base_logdir, args.method, 'saved_feat')
    if not os.path.exists(feat_save_path):
        os.mkdir(feat_save_path)

    for arg in vars(args):
        print('{}--{}'.format(arg, getattr(args, arg)))

    cudnn.deterministic = True
    cudnn.benchmark = False

    # np.random.seed(args.random_seed)
    # random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)

    # if args.split == 'target_outer':
    #     if args.data_type != 'origi':
    #         train_data_path = os.path.join(args.base_data_path, args.split,
    #                                        'CaseDE12K_class10_2048_K4_' + args.exp_dataset + '_' + args.data_type + '_source.h5')
    #         test_data_path = os.path.join(args.base_data_path, args.split,
    #                                       'CaseDE12K_class10_2048_K4_' + args.exp_dataset + '_' + args.data_type + '_target.h5')
    #         assert args.data_size == 1024
    #
    #     else:
    #         train_data_path = os.path.join(args.base_data_path, args.split,
    #                                        'CaseDE12K_class10_2048_' + args.exp_dataset + '_source.h5')
    #         test_data_path = os.path.join(args.base_data_path, args.split,
    #                                       'CaseDE12K_class10_2048_' + args.exp_dataset + '_target.h5')
    #         assert args.data_size == 2048
    # else:
    #     train_data_path = os.path.join(args.base_data_path, args.split,
    #                                    'CaseDE12K_class10_2048_' + args.exp_dataset + '_source.h5')
    #     test_data_path = os.path.join(args.base_data_path, args.split,
    #                                   'CaseDE12K_class10_2048_' + args.exp_dataset + '_target.h5')
    #     assert args.data_size == 2048

    train_data_path = os.path.join(args.base_data_path, 'PHM2009_dataset_{}_helical_C3_1_2ST_out.h5'.format(args.source_dataset))
    test_data_path = os.path.join(args.base_data_path, 'PHM2009_dataset_{}_helical_C3_1_2ST_out.h5'.format(args.target_dataset))

    if args.data_type == 'origi':
        assert args.data_size == 6600
        assert args.conv_final_size == 25
    elif args.data_type == 'fft':
        assert args.data_size == 3300
        assert args.conv_final_size == 12
    else:
        raise ValueError()

    if args.method == 'matching_network':

        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader


        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                 n_way=args.test_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_test,
                                                 data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        model = MatchingNetwork(args=args)

        for name, _ in model.named_parameters():
            print(name)

        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.train(True)
            model.train_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.train(False)
                test_accu, test_ci = model.test_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print('##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print('##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci, best_test_epo))
        print('###########################################################################')

    elif args.method == 'matching_network_feat_space':


        data_loader_pretrain_class = SimpleDataLoader(batch_size=args.pretrain_batchsize,
                                                      data_file=train_data_path,
                                                      data_type=args.data_type)

        data_loader_pretrain = data_loader_pretrain_class.data_loader

        
        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader


        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                 n_way=args.test_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_test,
                                                data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader



        model = MatchingNetwork_feat_space(args=args)

        for name, _ in model.named_parameters():
            print(name)

        # we first train the backbone by classify loss
        # Max num epo is set, and we stop the train if train loss stops to increase in 15 epochs

        if args.backbone_pretrain_path is '':
            # we should pretrain the backbone if pretrained path is not given
            # else the pretrain path is given, and it is set in the definition of the model class, so no action is needed
            best_pretrain_loss, best_pretrain_epo = 10000000000, -1
            best_pretrain_accu = -1
            num_not_improve = 0

            ############## set random seed for reproducible results####################
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)

            for epo in range(args.pretrain_epochs):
                model.backbone.train(True)
                model.pretrain_classify_model.train(True)

                mean_loss, mean_accu = model.train_classify_iter(data_loader_pretrain, epo)

                if mean_loss < best_pretrain_loss:
                    best_pretrain_loss = mean_loss
                    best_pretrain_epo = epo
                    best_pretrain_accu = mean_accu
                    # save best pretrain model
                    save_path = os.path.join(model_save_path, 'best_pretrain_classify_model.pth')
                    torch.save(model.pretrain_classify_model.state_dict(), save_path)

                    save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
                    torch.save(model.backbone.state_dict(), save_path)

                    num_not_improve = 0
                else:
                    num_not_improve += 1


                print('#####Pretrain best loss is {}, best accu is {} at epoch {}.'.format(best_pretrain_loss, best_pretrain_accu, best_pretrain_epo))

                if num_not_improve == 15:
                    break

            # end training, we load best pretrain model
            save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
            model.backbone.load_state_dict(torch.load(save_path))


        # after pretrain, we fix the backbone as a feature extractor
        model.backbone.train(False)

        # next we perform episodic training
        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)   # when val we set random seed to the same one to ensure the val tasks remains the same

        test_accu, test_ci = model.test_backbone_knn_iter(data_loader_test, -1, type='Test')
        print('##################Use feature knn, test result: {:.3f}+-{:.3f}.##################'.format(test_accu, test_ci))

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)   # when val we set random seed to the same one to ensure the val tasks remains the same

        model.metric_backbone.train(False)
        test_accu, test_ci = model.test_metric_iter(data_loader_test, -1, type='Test')

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.metric_backbone.train(True)
            model.train_metric_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.metric_backbone.train(False)
                test_accu, test_ci = model.test_metric_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print('##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print('##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci, best_test_epo))
        print('###########################################################################')

        if args.save_feat:
            model.load_state_dict(torch.load(save_path))
            print('Saving train features....')
            data_loader_train_class = SimpleDataLoader(batch_size=args.pretrain_batchsize, data_file=train_data_path, data_type=args.data_type)
            data_loader_train = data_loader_train_class.data_loader
            model.save_feature(data_loader_train, feat_save_path, 80, 'Train')
            print('Done')

            print('Saving test features....')
            data_loader_test_class = SimpleDataLoader(batch_size=args.pretrain_batchsize, data_file=test_data_path, data_type=args.data_type)
            data_loader_test = data_loader_test_class.data_loader
            model.save_feature(data_loader_test, feat_save_path, 80, 'Test')
            print('Done')

    elif args.method == 'matching_network_pretrain':

        data_loader_pretrain_class = SimpleDataLoader(batch_size=args.pretrain_batchsize,
                                                      data_file=train_data_path,
                                                      data_type=args.data_type)

        data_loader_pretrain = data_loader_pretrain_class.data_loader

        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader

        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                n_way=args.test_n_way,
                                                n_support=args.n_shot,
                                                n_query=args.n_query,
                                                n_eposide=args.episodes_each_epoch_test,
                                                data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        model = MatchingNetwork_pretrain(args=args)

        for name, _ in model.named_parameters():
            print(name)

        # we first train the backbone by classify loss
        # Max num epo is set, and we stop the train if train loss stops to increase in 5 epochs

        if args.backbone_pretrain_path is '':
            # we should pretrain the backbone if pretrained path is not given
            # else the pretrain path is given, and it is set in the definition of the model class, so no action is needed
            best_pretrain_loss, best_pretrain_epo = 10000000000, -1
            best_pretrain_accu = -1
            num_not_improve = 0

            ############## set random seed for reproducible results####################
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)

            for epo in range(args.pretrain_epochs):
                model.backbone.train(True)
                model.pretrain_classify_model.train(True)

                mean_loss, mean_accu = model.train_classify_iter(data_loader_pretrain, epo)

                if mean_loss < best_pretrain_loss:
                    best_pretrain_loss = mean_loss
                    best_pretrain_epo = epo
                    best_pretrain_accu = mean_accu
                    # save best pretrain model
                    save_path = os.path.join(model_save_path, 'best_pretrain_classify_model.pth')
                    torch.save(model.pretrain_classify_model.state_dict(), save_path)

                    save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
                    torch.save(model.backbone.state_dict(), save_path)

                    num_not_improve = 0
                else:
                    num_not_improve += 1

                print('#####Pretrain best loss is {}, best accu is {} at epoch {}.'.format(best_pretrain_loss,
                                                                                           best_pretrain_accu,
                                                                                           best_pretrain_epo))

                if num_not_improve == 15:
                    break

            # end training, we load best pretrain model
            save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
            model.backbone.load_state_dict(torch.load(save_path))

        # after pretrain, we finetune backbone with episodic training
        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.backbone.train(True)
            model.train_metric_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.backbone.train(False)
                test_accu, test_ci = model.test_metric_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci,
                                                                                        best_test_epo))
        print('###########################################################################')

    elif args.method == 'proto_network':

        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader


        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                 n_way=args.test_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_test,
                                                 data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        model = ProtoNetwork(args=args)

        for name, _ in model.named_parameters():
            print(name)

        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.train(True)
            model.train_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(
                    args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.train(False)
                test_accu, test_ci = model.test_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci,
                                                                                        best_test_epo))
        print('###########################################################################')

    elif args.method == 'proto_network_feat_space':

        data_loader_pretrain_class = SimpleDataLoader(batch_size=args.pretrain_batchsize,
                                                      data_file=train_data_path,
                                                      data_type=args.data_type)

        data_loader_pretrain = data_loader_pretrain_class.data_loader

        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader

        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                n_way=args.test_n_way,
                                                n_support=args.n_shot,
                                                n_query=args.n_query,
                                                n_eposide=args.episodes_each_epoch_test,
                                                data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        model = ProtoNetwork_feat_space(args=args)

        for name, _ in model.named_parameters():
            print(name)

        # we first train the backbone by classify loss
        # Max num epo is set, and we stop the train if train loss stops to increase in 15 epochs

        if args.backbone_pretrain_path is '':
            # we should pretrain the backbone if pretrained path is not given
            # else the pretrain path is given, and it is set in the definition of the model class, so no action is needed
            best_pretrain_loss, best_pretrain_epo = 10000000000, -1
            best_pretrain_accu = -1
            num_not_improve = 0

            ############## set random seed for reproducible results####################
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)

            for epo in range(args.pretrain_epochs):
                model.backbone.train(True)
                model.pretrain_classify_model.train(True)

                mean_loss, mean_accu = model.train_classify_iter(data_loader_pretrain, epo)

                if mean_loss < best_pretrain_loss:
                    best_pretrain_loss = mean_loss
                    best_pretrain_epo = epo
                    best_pretrain_accu = mean_accu
                    # save best pretrain model
                    save_path = os.path.join(model_save_path, 'best_pretrain_classify_model.pth')
                    torch.save(model.pretrain_classify_model.state_dict(), save_path)

                    save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
                    torch.save(model.backbone.state_dict(), save_path)

                    num_not_improve = 0
                else:
                    num_not_improve += 1

                print('#####Pretrain best loss is {}, best accu is {} at epoch {}.'.format(best_pretrain_loss,
                                                                                           best_pretrain_accu,
                                                                                           best_pretrain_epo))

                if num_not_improve == 15:
                    break

            # end training, we load best pretrain model
            save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
            model.backbone.load_state_dict(torch.load(save_path))

        # after pretrain, we fix the backbone as a feature extractor
        model.backbone.train(False)

        # next we perform episodic training
        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(
            args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

        model.metric_backbone.train(False)
        test_accu, test_ci = model.test_metric_iter(data_loader_test, -1, type='Test')

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.metric_backbone.train(True)
            model.train_metric_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(
                    args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.metric_backbone.train(False)
                test_accu, test_ci = model.test_metric_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci,
                                                                                        best_test_epo))
        print('###########################################################################')

        if args.save_feat:
            model.load_state_dict(torch.load(save_path))
            print('Saving train features....')
            data_loader_train_class = SimpleDataLoader(batch_size=args.pretrain_batchsize, data_file=train_data_path, data_type=args.data_type)
            data_loader_train = data_loader_train_class.data_loader
            model.save_feature(data_loader_train, feat_save_path, 80, 'Train')
            print('Done')

            print('Saving test features....')
            data_loader_test_class = SimpleDataLoader(batch_size=args.pretrain_batchsize, data_file=test_data_path, data_type=args.data_type)
            data_loader_test = data_loader_test_class.data_loader
            model.save_feature(data_loader_test, feat_save_path, 80, 'Test')
            print('Done')

    elif args.method == 'proto_network_pretrain':

        data_loader_pretrain_class = SimpleDataLoader(batch_size=args.pretrain_batchsize,
                                                      data_file=train_data_path,
                                                      data_type=args.data_type)

        data_loader_pretrain = data_loader_pretrain_class.data_loader

        data_loader_train_class = MetaDataLoader(data_file=train_data_path,
                                                 n_way=args.train_n_way,
                                                 n_support=args.n_shot,
                                                 n_query=args.n_query,
                                                 n_eposide=args.episodes_each_epoch_train,
                                                 data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader

        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                n_way=args.test_n_way,
                                                n_support=args.n_shot,
                                                n_query=args.n_query,
                                                n_eposide=args.episodes_each_epoch_test,
                                                data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        model = ProtoNetwork_pretrain(args=args)

        for name, _ in model.named_parameters():
            print(name)

        # we first train the backbone by classify loss
        # Max num epo is set, and we stop the train if train loss stops to increase in 5 epochs

        if args.backbone_pretrain_path is '':
            # we should pretrain the backbone if pretrained path is not given
            # else the pretrain path is given, and it is set in the definition of the model class, so no action is needed
            best_pretrain_loss, best_pretrain_epo = 10000000000, -1
            best_pretrain_accu = -1
            num_not_improve = 0

            ############## set random seed for reproducible results####################
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)

            for epo in range(args.pretrain_epochs):
                model.backbone.train(True)
                model.pretrain_classify_model.train(True)

                mean_loss, mean_accu = model.train_classify_iter(data_loader_pretrain, epo)

                if mean_loss < best_pretrain_loss:
                    best_pretrain_loss = mean_loss
                    best_pretrain_epo = epo
                    best_pretrain_accu = mean_accu
                    # save best pretrain model
                    save_path = os.path.join(model_save_path, 'best_pretrain_classify_model.pth')
                    torch.save(model.pretrain_classify_model.state_dict(), save_path)

                    save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
                    torch.save(model.backbone.state_dict(), save_path)

                    num_not_improve = 0
                else:
                    num_not_improve += 1

                print('#####Pretrain best loss is {}, best accu is {} at epoch {}.'.format(best_pretrain_loss,
                                                                                           best_pretrain_accu,
                                                                                           best_pretrain_epo))

                if num_not_improve == 15:
                    break

            # end training, we load best pretrain model
            save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
            model.backbone.load_state_dict(torch.load(save_path))

        # after pretrain, we finetune backbone with episodic training
        best_test_accu, best_test_ci = -1, -1
        best_test_epo = -1
        num_not_improve = 0

        for epo in range(args.meta_epochs):

            ############## set random seed for reproducible results####################
            np.random.seed(epo)
            random.seed(epo)
            torch.manual_seed(epo)
            torch.cuda.manual_seed(epo)

            model.backbone.train(True)
            model.train_metric_iter(data_loader_train, epo)

            # perform val every some epochs
            if epo % args.meta_val_interval == 0:

                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                torch.manual_seed(args.random_seed)
                torch.cuda.manual_seed(
                    args.random_seed)  # when val we set random seed to the same one to ensure the val tasks remains the same

                model.backbone.train(False)
                test_accu, test_ci = model.test_metric_iter(data_loader_test, epo, type='Test')

                # lr_scheduler.step(val_accu)  # adjust lr

                if test_accu >= best_test_accu:
                    num_not_improve = 0
                    best_test_epo = epo

                    best_test_accu = test_accu
                    best_test_ci = test_ci
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                    # save best val model
                    save_path = os.path.join(model_save_path, 'best_test_model.pth')
                    torch.save(model.state_dict(), save_path)

                else:
                    num_not_improve += 1
                    print(
                        '##################Current best test result: {:.3f}+-{:.3f} at epoch {}.##################'.format(
                            best_test_accu, best_test_ci, best_test_epo))

                # if num_not_improve > 20:  # if val accu not improved in 20 times, we stop training and report final results
                #     break
        print('###########################################################################')
        print('Training stopped. Best test result: {:.3f}+-{:.3f} at epoch {}. '.format(best_test_accu, best_test_ci,
                                                                                        best_test_epo))
        print('###########################################################################')

    else:

        data_loader_train_class = SimpleDataLoader(batch_size=args.pretrain_batchsize,
                                                   data_file=train_data_path,
                                                   data_type=args.data_type)

        data_loader_train = data_loader_train_class.data_loader


        data_loader_test_class = MetaDataLoader(data_file=test_data_path,
                                                n_way=args.test_n_way,
                                                n_support=args.n_shot,
                                                n_query=args.n_query,
                                                n_eposide=args.episodes_each_epoch_test,
                                                data_type=args.data_type)

        data_loader_test = data_loader_test_class.data_loader

        if args.method == 'finetune_last':
            model = Baseline_FinetuneLast(args)
        elif args.method == 'finetune_whole':
            model = Baseline_FinetuneWhole(args)
        elif args.method == 'feature_knn':
            model = Baseline_FeatureKnn(args)
        elif args.method == 'feature_knn_proto':
            model = Baseline_FeatureKnn_Proto(args)
        else:
            raise ValueError()

        for name, _ in model.named_parameters():
            print(name)

        # we first train the backbone by classify loss
        # Max num epo is set, and we stop the train if train loss stops to increase in 15 epochs

        if args.backbone_pretrain_path is '':
            # we should pretrain the backbone if pretrained path is not given
            # else the pretrain path is given, and it is set in the definition of the model class, so no action is needed
            best_pretrain_loss, best_pretrain_epo = 10000000000, -1
            best_pretrain_accu = -1
            num_not_improve = 0

            ############## set random seed for reproducible results####################
            np.random.seed(args.random_seed)
            random.seed(args.random_seed)
            torch.manual_seed(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)

            for epo in range(args.pretrain_epochs):
                model.backbone.train(True)
                model.pretrain_classify_model.train(True)

                mean_loss, mean_accu = model.train_classify_iter(data_loader_train, epo)

                if mean_loss < best_pretrain_loss:
                    best_pretrain_loss = mean_loss
                    best_pretrain_epo = epo
                    best_pretrain_accu = mean_accu
                    # save best pretrain model
                    save_path = os.path.join(model_save_path, 'best_pretrain_classify_model.pth')
                    torch.save(model.pretrain_classify_model.state_dict(), save_path)

                    save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
                    torch.save(model.backbone.state_dict(), save_path)

                    num_not_improve = 0
                else:
                    num_not_improve += 1


                print('#####Pretrain best loss is {}, best accu is {} at epoch {}.'.format(best_pretrain_loss, best_pretrain_accu, best_pretrain_epo))

                if num_not_improve == 15:
                    break

            # end training, we load best pretrain model
            save_path = os.path.join(model_save_path, 'best_pretrain_backbone.pth')
            model.backbone.load_state_dict(torch.load(save_path))


        # then we fix backbone and perform next step

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed) # when val we set random seed to the same one to ensure the val tasks remains the same

        model.backbone.train(False)

        if args.method == 'finetune_whole':
            test_accu, test_ci = model.test_iter_meta_finetune(data_loader_test, epo, result_subdir)
        else:
            test_accu, test_ci = model.test_iter_meta_finetune(data_loader_test, epo)

        print('###########################################################################')
        print('Baseline method {} test result: {:.3f}+-{:.3f}. '.format(args.method, test_accu, test_ci))
        print('###########################################################################')

