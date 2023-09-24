import logging
import numpy as np
import time
import torch
import torch.nn.functional as F
from collections import Counter
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

from common import to_categorical
from losses import TripletMSELoss
from losses import HiDistanceXentLoss
from samplers import ProportionalClassSampler
from samplers import HalfSampler
from samplers import TripletSampler
from utils import AverageMeter
from utils import save_model
from utils import adjust_learning_rate

def pseudo_loss(args, encoder, X_train, y_train, y_train_binary, \
                X_test, y_test_pred, test_offset, total_epochs):
    # contruct the dataset loader
    X_tensor = torch.from_numpy(np.vstack((X_train, X_test))).float()
    # y has a mix of real labels and predicted pseudo labels.
    # y = np.concatenate((y_train, y_test_pred), axis=0)
    # y has binary labels
    # logging.debug(f'y_train_binary, {y_train_binary}')
    # logging.debug(f'y_test_pred, {y_test_pred}')
    y = np.concatenate((y_train_binary, y_test_pred), axis=0)
    # logging.debug(f'y, {y}')
    # logging.debug(f'y.shape, {y.shape}')

    # y_tensor is used for computing similarity matrix => supcon loss
    y_tensor = torch.from_numpy(y)
    
    device = (torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu'))
    encoder = encoder.to(device)

    # DEBUG buggy version
    # y_bin_pred = torch.zeros(X_test.shape[0], 2)
    # y_bin_train_cat = torch.from_numpy(to_categorical(y_train_binary)).float()
    # y_bin_cat_tensor = torch.cat((y_bin_train_cat, y_bin_pred), dim = 0)

    # correct version
    y_bin_cat_tensor = torch.from_numpy(to_categorical(y, num_classes=2)).float()
    # logging.debug(f'y_bin_cat_tensor.shape, {y_bin_cat_tensor.shape}')

    split_tensor = torch.zeros(X_tensor.shape[0]).int()
    split_tensor[test_offset:] = 1
    index_tensor = torch.from_numpy(np.arange(y.shape[0]))

    all_data = TensorDataset(X_tensor, y_tensor, y_bin_cat_tensor, index_tensor, split_tensor)

    if args.sampler == 'mperclass':
        bsize = args.sample_per_class * len(np.unique(y))
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = MPerClassSampler(y, args.sample_per_class))
    elif args.sampler == 'proportional':
        if args.bsize is None:
            bsize = args.min_per_class * len(np.unique(y))
        else:
            bsize = args.bsize
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = ProportionalClassSampler(y, args.min_per_class, bsize))
    elif args.sampler == 'half':
        # default pseudo loss batch size is the same as the training batch size
        if args.plb == None:
            bsize = args.bsize
        else:
            bsize = args.plb
        data_loader = DataLoader(dataset=all_data, batch_size=bsize, \
            sampler = HalfSampler(y, bsize))
    elif args.sampler == 'random':
        bsize = args.bsize
        train_loader = DataLoader(dataset=all_data, batch_size=bsize, shuffle=True)
    else:
        raise Exception('Need to add a sampler here.')
    
    sample_num = y.shape[0]
    sum_loss = np.zeros([sample_num])
    cur_sample_loss = np.zeros([sample_num])
    for epoch in range(1, total_epochs + 1):
        # pseudo_loss goes through one epoch, loss for all samples
        time1 = time.time()
        sample_loss = pseudo_loss_one_epoch(args, encoder, data_loader, sample_num, epoch)
        time2 = time.time()
        if args.sample_reduce == 'mean':
            sum_loss += sample_loss
            # average the loss per sample, including both train and test
            cur_sample_loss = sum_loss / epoch
            # only print test sample cur_sample_loss
            logging.info('epoch {}, b {}, total time {:.2f}, (sorted avg loss)[:50] {}'.format(epoch, bsize, time2 - time1, sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
        else:
            # args.sample_reduce == 'max':
            cur_sample_loss = np.maximum(cur_sample_loss, sample_loss)
            # only print test sample cur_sample_loss
            logging.info('epoch {}, b {}, total time {:.2f}, (sorted max loss)[:50] {}'.format(epoch, bsize, time2 - time1, sorted(cur_sample_loss[test_offset:], reverse=True)[:50]))
    return cur_sample_loss

def pseudo_loss_one_epoch(args, encoder, data_loader, sample_num, epoch):
    """
    measure one epoch of pseudo loss for train + test samples.
    default data points number in an epoch: length_before_new_iter=100000
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    
    select_count = torch.zeros(sample_num, dtype=torch.float64)
    total_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'mean'
    sample_avg_loss = torch.zeros(sample_num, dtype=torch.float64)
    # if args.sample_reduce == 'max'
    sample_max_loss = torch.zeros(sample_num, dtype=torch.float64)

    pos_max_sim = torch.zeros(sample_num, dtype=torch.float64)
    neg_max_sim = torch.zeros(sample_num, dtype=torch.float64)

    idx = 0
    # average the loss for each index in batch_indices
    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    
    for idx, (x_batch, y_batch, y_bin_batch, batch_indices, split_tensor) in enumerate(data_loader):
        data_time.update(time.time() - end)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_bin_batch = y_bin_batch.to(device)
        
        if args.loss_func == 'triplet-mse':
            features, decoded = encoder(x_batch)

            TripletMSE = TripletMSELoss(reduce = args.reduce).to(device)
            loss, _, _ = TripletMSE(args.cae_lambda, \
                                x_batch, decoded, features, labels=y_batch, \
                                margin = args.margin, \
                                split = split_tensor)
            loss = loss.to('cpu').detach()
        elif args.loss_func == 'hi-dist-xent':
            _, features, y_pred = encoder(x_batch)
            HiDistanceXent = HiDistanceXentLoss(reduce = args.reduce).to(device)
            loss, _, _ = HiDistanceXent(args.xent_lambda, 
                                    y_pred, y_bin_batch,
                                    features, labels=y_batch,
                                    split = split_tensor)
            loss = loss.to('cpu').detach()
        else:
            raise Exception(f'pseudo loss for {args.loss_func} not implemented.')
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        select_count[batch_indices] = torch.add(select_count[batch_indices], 1)
        non_select_count = sample_num - torch.count_nonzero(select_count).item()
        # # update the loss values for batch_indices
        if args.sample_reduce == 'mean':
            total_loss[batch_indices] = torch.add(total_loss[batch_indices], loss)

        if args.sample_reduce == 'max':
            sample_max_loss[batch_indices] = torch.maximum(sample_max_loss[batch_indices], loss)
            
    # sample average loss
    if args.sample_reduce == 'mean':
        sample_avg_loss = torch.div(total_loss, select_count)
        return sample_avg_loss.numpy()
    else:
        # args.sample_reduce == 'max':
        return sample_max_loss.numpy()

def train_encoder(args, encoder, X_train, y_train, y_train_binary,
                optimizer, total_epochs, model_path,
                weight = None, upsample = None, adjust = False, warm = False,
                save_best_loss = False,
                save_snapshot = False,
                pl_pretrain = False):
    # construct the dataset loader
    # y_train is multi-class, y_train_binary is binary class
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).type(torch.int64)
    y_train_binary_cat_tensor = torch.from_numpy(to_categorical(y_train_binary)).float()
    if weight is None:
        weight_tensor = torch.ones(X_train.shape[0])
    else:
        weight_tensor = torch.from_numpy(weight).float()
    train_data = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_cat_tensor, weight_tensor)

    # compute batch size if it is not specified
    if args.sampler == 'mperclass':
        bsize = args.sample_per_class * len(np.unique(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = MPerClassSampler(y_train, args.sample_per_class))
    elif args.sampler == 'proportional':
        if args.bsize is None:
            bsize = args.min_per_class * len(np.unique(y_train))
        else:
            bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = ProportionalClassSampler(y_train, args.min_per_class, bsize))
    elif args.sampler == 'half':
        bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
            sampler = HalfSampler(y_train, bsize, upsample = upsample))
    elif args.sampler == 'random':
        bsize = args.bsize
        train_loader = DataLoader(dataset=train_data, batch_size=bsize, shuffle=True)
    else:
        raise Exception(f'Sampler {args.sampler} not implemented yet.')
    best_loss = np.inf
    for epoch in range(1, total_epochs + 1):
        if adjust == True:
            # only adjust learning rate when training the initial model.
            # retraining assigns sample weight so we are not adjust learning rates.
            new_lr = adjust_learning_rate(args, optimizer, epoch, warm = warm)
        else:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr']
                break
        
        # train one epoch
        time1 = time.time()
        if pl_pretrain == False:
            loss = train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch)
        else:
            loss = pl_train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch)
        time2 = time.time()
        logging.info('epoch {}, b {}, lr {}, loss {}, total time {:.2f}'.format(epoch, bsize, new_lr, loss, time2 - time1))

        if epoch >= total_epochs - 10:
            if save_best_loss == True:
                if loss < best_loss:
                    best_loss = loss
                    logging.info(f'Saving the best loss {loss} model from epoch {epoch}...')
                    save_model(encoder, optimizer, args, args.epochs, model_path)
    
        if save_snapshot == True and epoch % 50 == 0:
            save_path = model_path.replace("e%s" % total_epochs, "e%d" % epoch)
            logging.info(f'Saving the model from epoch {epoch} loss {loss} at {save_path}...')
            save_model(encoder, optimizer, args, args.epochs, save_path)
    return

def train_encoder_one_epoch(args, encoder, train_loader, optimizer, epoch):
    """ Train one epoch for the model """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    supcon_losses = AverageMeter()
    mse_losses = AverageMeter()
    xent_losses = AverageMeter()
    xent_multi_losses = AverageMeter()
    xent_bin_losses = AverageMeter()
    end = time.time()

    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    encoder = encoder.to(device)

    idx = 0
    for idx, (x_batch, y_batch, y_bin_batch, weight_batch) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_bin_batch = y_bin_batch.to(device)
        weight_batch = weight_batch.to(device)

        # DEBUG
        # logging.debug(Counter(y_batch.cpu().detach().numpy()))
        # logging.debug(y_batch.cpu().detach().numpy())

        bsz = y_batch.shape[0]

        if args.loss_func == 'triplet-mse':
            features, decoded = encoder(x_batch)

            TripletMSE = TripletMSELoss().cuda()
            loss, supcon_loss, mse_loss = TripletMSE(args.cae_lambda, \
                                x_batch, decoded, features, labels=y_batch, \
                                margin = args.margin, \
                                weight = weight_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            mse_losses.update(mse_loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) % args.display_interval == 0:
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'triplet {supcon.val:.4f} ({supcon.avg:.4f})  '
                    'mse {mse.val:.4f} ({mse.avg:.4f})  '.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    supcon=supcon_losses,
                    mse=mse_losses))
        
        elif args.loss_func == 'hi-dist-xent':
            _, cur_f, y_pred = encoder(x_batch)
        
            # features: hidden vector of shape [bsz, n_feature_dim].
            features = cur_f

            # Our own version of the supervised contrastive learning loss
            HiDistanceXent = HiDistanceXentLoss().cuda()
            loss, supcon_loss, xent_loss = HiDistanceXent(args.xent_lambda, \
                                            y_pred, y_bin_batch, \
                                            features, labels = y_batch, \
                                            margin = args.margin, \
                                            weight = weight_batch)
            
            # update metric
            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            xent_losses.update(xent_loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info, print every display_interval batches.
            if (idx + 1) % args.display_interval == 0:
                logging.info('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'loss {loss.val:.5f} ({loss.avg:.5f})  '
                    'hidist {supcon.val:.5f} ({supcon.avg:.5f})  '
                    'xent {xent.val:.5f} ({xent.avg:.5f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, supcon=supcon_losses,
                    xent=xent_losses))

        else:
            raise Exception(f'The loss function {args.loss_func} for model ' \
                f'{args.encoder} is not supported yet.')
        
    return losses.avg

def train_classifier(args, classifier, X_train, y_train, y_train_binary, 
                    optimizer, total_epochs, model_path,
                    weight = None, multi = False,
                    save_best_loss = False,
                    save_snapshot = False):
    # contruct the dataset loader
    # y_train is multi-class, y_train_binary is binary class
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    if weight is None:
        weight_tensor = torch.ones(X_train_tensor.shape[0])
    else:
        weight_tensor = torch.from_numpy(weight).float()

    if multi == False:
        y_train_binary_cat_tensor = torch.from_numpy(to_categorical(y_train_binary)).float()
        train_data = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_cat_tensor, weight_tensor)
        train_loader = DataLoader(dataset=train_data, batch_size=args.mlp_batch_size, shuffle=True)
    else:
        y_train_cat_tensor = torch.from_numpy(to_categorical(y_train)).float()
        train_data = TensorDataset(X_train_tensor, y_train_tensor, y_train_cat_tensor, weight_tensor)
        # train_loader = DataLoader(dataset=train_data, batch_size=args.mlp_batch_size, shuffle=True)
        if args.sampler == 'mperclass':
            bsize = args.sample_per_class * len(np.unique(y_train))
            train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
                sampler = MPerClassSampler(y_train, args.sample_per_class))
        elif args.sampler == 'proportional':
            bsize = args.mlp_batch_size
            train_loader = DataLoader(dataset=train_data, batch_size=bsize, \
                sampler = ProportionalClassSampler(y_train, args.min_per_class, bsize))
        elif args.sampler == 'random':
            bsize = args.mlp_batch_size
            train_loader = DataLoader(dataset=train_data, batch_size=bsize, shuffle=True)
        else:
            raise Exception(f'Sampler {args.sampler} not implemented yet.')

    best_loss = np.inf
    for epoch in range(1, total_epochs + 1):
        # train one epoch
        time1 = time.time()
        loss = train_classifier_one_epoch(args, classifier, train_loader,
                                        optimizer, epoch, multi = multi)
        time2 = time.time()
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr']
            break
        logging.info('epoch {}, b {}, lr {}, loss {}, total time {:.2f}'.format(epoch, args.mlp_batch_size, new_lr, loss, time2 - time1))

        if save_best_loss == True:
            if loss < best_loss:
                best_loss = loss
                logging.info(f'Saving the best loss {loss} model from epoch {epoch}...')
                save_model(classifier, optimizer, args, args.epochs, model_path)
        
        if save_snapshot == True and epoch % 25 == 0:
            save_path = model_path.replace("e%s" % total_epochs, "e%d" % epoch)
            logging.info(f'Saving the model from epoch {epoch} loss {loss} at {save_path}...')
            save_model(classifier, optimizer, args, args.epochs, save_path)
    
    return

def train_classifier_one_epoch(args, classifier, train_loader, optimizer, epoch, multi = False):
    """ Train one epoch for the model """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    device = (torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu'))
    classifier = classifier.to(device)

    idx = 0
    for idx, (x_batch, y_batch, y_cat_batch, weight_batch) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_cat_batch = y_cat_batch.to(device)
        weight_batch = weight_batch.to(device)

        # DEBUG
        # print(Counter(y_batch.cpu().detach().numpy()))
        # print(y_cat_batch.cpu().detach().numpy())

        bsz = y_batch.shape[0]

        if multi == False:
            y_pred = classifier.predict_proba(x_batch)[:, 1]
            # logging.debug(f'y_pred {y_pred}')
            target = y_cat_batch[:, 1]
            # logging.debug(f'target {target}')
            loss_ele = torch.nn.functional.binary_cross_entropy(y_pred, target, reduction = 'none')
            loss = loss_ele * weight_batch
            # logging.debug(f'loss: {loss}')
            loss = loss.mean()
            # logging.debug(f'loss.mean(): {loss}')
        else:
            y_multi_pred = classifier.predict_proba(x_batch)
            # logging.info(f'y_multi_pred {y_multi_pred}')
            # logging.info(f'y_batch {y_batch}')
            # loss_ele = torch.nn.functional.cross_entropy(y_multi_pred, y_batch, reduction = 'none')
            ## loss = weight_batch.view(-1, 1) * loss_ele
            # loss = weight_batch * loss_ele
            # loss = loss.mean()
            loss = torch.nn.functional.cross_entropy(y_multi_pred, y_batch)
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info, print every mlp_display_interval batches.
        if (idx + 1) % args.mlp_display_interval == 0:
            logging.info('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
                'loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time,
                loss=losses))
        
    return losses.avg
