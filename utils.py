import os
import sys
import logging
import argparse
import math
import numpy as np

def parse_args():
    """Parse the command line configuration for a particular run.
    
    Returns:
        argparse.Namespace -- a set of parsed arguments.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--data', help='The dataset to use.')
    p.add_argument('--mdate', help='Encoder model date to use.')
    p.add_argument('--clsdate', help='Classifier model date to use.')

    p.add_argument('--train', default=None, help='Train month. e.g., 2012-01')
    p.add_argument('--train_start', default=None, help='Train start month. e.g., 2012-01')
    p.add_argument('--train_end', default=None, help='Train end month. e.g., 2012-12')
    p.add_argument('--benign_zero', action='store_true', help='Whether assign benign class as the 0 label.')

    p.add_argument('--test_start', help='First test month.')
    p.add_argument('--test_end', help='Last test month.')

    # loss function choices
    p.add_argument('--loss_func', default='hi-dist-xent',
            choices=['triplet',
                'triplet-mse',
                'hi-dist-xent'],
            help='contrastive loss function choice.')

    # active learning parameters
    p.add_argument('--al', action='store_true', help='Whether to do active learning.')
    p.add_argument('--accumulate_data', action='store_true', help='Whether to accumulate test data from previous month, excluding the selected test samples')
    p.add_argument('--transcend', action='store_true', help='Use transcend score to help sampling')
    p.add_argument('--criteria', default='cred',
                   choices=['cred', 'conf', 'cred+conf'],
                   help='Transcendent: the p-values to threshold on.')
    p.add_argument('--ood', action='store_true', help='Use CAE OOD score to help sampling')
    p.add_argument('--local_pseudo_loss', action='store_true', help='Use local pseudo loss to select samples')
    p.add_argument('--reduce', type=str, choices=['none', 'max', 'mean'],
                    help='how to reduce the loss to compute the pseudo loss')
    p.add_argument('--sample_reduce', type=str, choices=['mean', 'max'],
                    help='how to reduce the loss per sample')

    p.add_argument('--unc', action='store_true', help='Uncertain sampling')
    p.add_argument('--rand', action='store_true', help='Random sampling')

    p.add_argument('--count', type=int, default=None, help='Sampling count')
    p.add_argument('--result', type=str, help='file name to generate MLP performance csv result.')

    # encoder model
    p.add_argument('--encoder', default=None, \
                    choices=['cae', 'enc', 'mlp', \
                            'simple-enc-mlp'], \
                    help='The encoder model to get embeddings of the input.')
    p.add_argument('--encoder-retrain', action='store_true',
                   help='Whether to train the encoder again.')
    p.add_argument('--cold-start', action='store_true',
                   help='Whether to retrain the encoder from scratch.')

    # classifier
    p.add_argument('-c', '--classifier', default='svm',
                   choices=['mlp', 'svm', 'gbdt', \
                            'simple-enc-mlp'],
                   help='The target classifier to use.')
    p.add_argument('--svm-c', default=1.0, type=float,
                   help='Regularization parameter for SVM.' \
                    'The strength of the regularization is inversely proportional to C.')
    p.add_argument('--max_depth', default=6, type=int,
                   help='GBDT: max_depth in the tree ensemble.')
    p.add_argument('--num_round', default=10, type=int,
                   help='GBDT: number of boosting rounds / trees.')
    p.add_argument('--eta', default=0.3, type=float,
                help='GBDT: learning rate.')
    
    # arguments for the SVM classifier.
    p.add_argument('--multi_class', action='store_true', help='train multi-class.')
    p.add_argument('--eval_multi', action='store_true', help='evaluate multi-class prediction performance.')
    
    # for debugging messages
    p.add_argument('--verbose', action='store_true',
                    help='whether to print the debugging logs.')
    
    # arguments for the Encoder Classifier model.
    p.add_argument('--enc-hidden',
                help='The hidden layers of the encoder, example: "512-128-32"')
    p.add_argument('--bsize', default=None, type=int,
                   help='Training batch size.')
    p.add_argument('--plb', default=None, type=int,
                   help='Pseudo loss batch size.')
    p.add_argument('--sample-per-class', default=2, type=int,
                   help='Number of samples for each class in a batch.')
    p.add_argument('--min-per-class', default=2, type=int,
                   help='Minimum number of samples for each class in a batch.')
    p.add_argument('--learning_rate', default=0.01, type=float,
                   help='Overall learning rate.')
    p.add_argument('--warm_learning_rate', default=0.001, type=float,
                   help='Warm start learning rate.')
    p.add_argument('--scheduler', default='step', type=str, choices=['step', 'cosine'],
                   help='Choosing the learning rate decay scheduler.')
    p.add_argument('--lr_decay_rate', type=float, default=1,
                        help='decay rate for learning rate')
    p.add_argument('--lr_decay_epochs', type=str, default='30,1000,30',
                        help='where to decay lr. start epoch, end epoch, step size.')
    p.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'],
                        help='Choosing an optimzer')
    p.add_argument('--al_optimizer', default=None, type=str, choices=['adam', 'sgd'],
                        help='Choosing an optimzer')
    p.add_argument('--epochs', default=250, type=int,
                   help='Training epochs.')
    p.add_argument('--al_epochs', default=50, type=int,
                   help='Active learning training epochs.')
    p.add_argument('--xent-lambda', default=1, type=float,
                   help='lambda to scale the binary cross entropy loss.')
    p.add_argument('--log_path', type=str,
                   help='log file name.')
    p.add_argument('--retrain-first', action='store_true',
                   help='Whether to retrain the first model.')
    p.add_argument('--sampler', type=str, choices=['mperclass', 'proportional', 'half',
                    'triplet', 'random'],
                   help='The sampler to sample batches.')
    p.add_argument('--snapshot', action='store_true',
                   help='Whether to save the model at every 50 epoch.')
    
    # arguments for the Autoencode + Classifier
    p.add_argument('--mse-lambda', default=1, type=float,
                help='lambda to scale the MSE loss.')
    
    # arguments for the Contrastive Autoencoder and drift detection (build on the samples of top 7 families for example)
    p.add_argument('--cae-hidden',
                   help='The hidden layers of the giant autoencoder, example: "512-128-32", \
                         which in drebin_new_7 would make the architecture as "1340-512-128-32-7"')
    p.add_argument('--cae-batch-size', default=64, type=int,
                   help='Contrastive Autoencoder batch_size, use a bigger size for larger training set \
                        (when training, one batch only has 64/2=32 samples, another 32 samples are used for comparison).')
    p.add_argument('--cae-lr', default=0.001, type=float,
                   help='Contrastive Autoencoder Adam learning rate.')
    p.add_argument('--cae-epochs', default=250, type=int,
                   help='Contrastive Autoencoder epochs.')
    p.add_argument('--cae-lambda', default=1e-1, type=float,
                   help='lambda in the loss function of contrastive autoencoder.')
    p.add_argument('--margin', default=10.0, type=float,
                    help='Maximum margins of dissimilar samples when training contrastive autoencoder.')
    p.add_argument('--display-interval', default=10, type=int,
                    help='Show logs about loss and other information every xxx epochs when training the encoder.')

    p.add_argument('--mad-threshold', default=3.5, type=float,
                    help='The threshold for MAD outlier detection, choose one from 2, 2.5, 3 or 3.5')
    
    # sub-arguments for the MLP classifier.
    p.add_argument('--cls-retrain', type=int, default=0, choices=[0, 1],
                   help='Whether to retrain the classifier.')
    p.add_argument('--cls-feat', type=str, default='input', choices=['encoded', 'input'],
                   help='input features for the classifier.')
    p.add_argument('--mlp-hidden',
                   help='The hidden layers of the MLP classifier, example: "100-30", which in drebin_new_7 case would make the architecture as 1340-100-30-7')
    p.add_argument('--mlp-batch-size', default=32, type=int,
                   help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float,
                   help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=50, type=int,
                   help='MLP classifier epochs.')
    p.add_argument('--mlp-warm-lr', default=0.001, type=float,
                   help='MLP classifier AL warm start learning rate.')
    p.add_argument('--mlp-warm-epochs', default=50, type=int,
                   help='MLP classifier AL warm start epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float,
                   help='MLP classifier Droput rate.')
    p.add_argument('--mlp-display-interval', default=300, type=int,
                    help='Show logs about loss and other information every xxx epochs when training contrastive autoencoder.')

    args = p.parse_args()

    return args


def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    """convert hidden layer arguments to the architecture of a model (list)

    Arguments:
        model_name {str} -- 'MLP' or 'Contrastive AE' or 'Encoder'.
        input_layer_num {int} -- The number of the features.
        hidden_layer_num {str} -- The '-' connected numbers indicating the number of neurons in hidden layers.
        output_layer_num {int} -- The number of the classes.

    Returns:
        [list] -- List represented model architecture.
    """
    try:
        if '-' not in hidden_layer_num:
            if model_name == 'MLP':
                dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
            else:
                dims = [input_layer_num, int(hidden_layer_num)]
        else:
            hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
            dims = [input_layer_num]
            for dim in hidden_layers:
                dims.append(dim)
            if model_name == 'MLP':
                dims.append(output_layer_num)
        logging.debug(f'{model_name} dims: {dims}')
    except:
        logging.error(f'get_model_dims {model_name}')
        sys.exit(-1)

    return dims

def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

def adjust_learning_rate(args, optimizer, epoch, warm = False):
    if warm == False:
        lr = args.learning_rate
    else:
        lr = args.warm_learning_rate
    # use the same learning rate scheduler in active learning and initial training
    if args.scheduler == 'cosine':
        # eta_min = 1e-11
        eta_min = 0
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    elif args.scheduler == 'step':
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    else:
        raise Exception('scheduler {args.scheduler} not supported yet.}')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

# From https://github.com/HobbitLong/SupContrast
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# From https://github.com/HobbitLong/SupContrast
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
