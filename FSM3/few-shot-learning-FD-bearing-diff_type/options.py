import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot FD settings')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to train")
    parser.add_argument('--random_seed', default=10, type=int, help='random seed in experiments, default is 10')
    parser.add_argument('--base_data_path', default='/data_disk/dataset/FD/bearing_data/2048/', help='base path keeping image data')

    parser.add_argument('--exp_dataset', default='dataset_3', help='dataset_0/1/2/3')
    parser.add_argument('--split', default='target_outer', help='class10, target_4-7-10, target_ball, target_inner, target_outer')
    parser.add_argument('--data_type', default='origi', help='only for target_outer, fft or h_fft or origi')
    parser.add_argument('--data_size', default=2048, help='1024 for fft, 2048 for default')

    parser.add_argument('--backbone_out_dim', default=100, type=int, help='feature dim')
    parser.add_argument('--in_channels', default=1, type=int, help='num of input channels')

    parser.add_argument('--base_logdir', default='/data_disk/deeplearning/few-shot-learning-FD/exp_logs/', help='folder to keep all running files')
    parser.add_argument('--method', default='proto_network_feat_space', help='finetune_last/finetune_whole/feature_knn/matching_network/matching_network_feat_space/matching_network_pretrain'
                                                                '/feature_knn_proto/proto_network/proto_network_feat_space/proto_network_pretrain')

    # few-shot task setting
    parser.add_argument('--train_n_way', default=7, type=int, help='class num to classify for training')
    parser.add_argument('--test_n_way', default=3, type=int, help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int, help='number of labeled support data in each class')
    parser.add_argument('--n_query', default=25, type=int, help='number of query(test) data in each class')

    parser.add_argument('--meta_epochs', default=100, type=int, help='total number of meta training epochs')
    parser.add_argument('--episodes_each_epoch_train', default=100, type=int, help='Number of episodes in each meta train epoch')
    parser.add_argument('--episodes_each_epoch_test', default=600, type=int, help='Number of episodes in each meta test epoch')
    parser.add_argument('--meta_val_interval', default=1, type=int, help='Meta val interval')

    # settings of pretrain normal CNN
    parser.add_argument('--pretrain_source_num_classes', default=7, type=int, help='total number of classes')
    parser.add_argument('--pretrain_target_num_classes', default=3, type=int, help='total number of classes')
    parser.add_argument('--pretrain_epochs', default=50, type=int, help='total number of pretrain epochs')
    parser.add_argument('--pretrain_batchsize', default=16, type=int, help='batch size of pretrain epochs')
    parser.add_argument('--pretrain_save_freq', default=10, type=int, help='save interval')

    parser.add_argument('--backbone_pretrain_path', default='', type=str, help='pretrain backbone path')  # /data_disk/deeplearning/few-shot-learning-FD/exp_logs/matching_network_pretrain/TRAIN/exp__003/saved_model/best_pretrain_backbone.pth

    parser.add_argument('--pretrain_finetune_steps', default=100, type=int, help='pretrain_finetune_steps, for finetune whole model set to 100, for finetune last, set to 150')


    return parser.parse_args()


