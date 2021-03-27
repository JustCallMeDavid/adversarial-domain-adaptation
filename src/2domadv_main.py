from Util import *
from Models import *
import argparse
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import cycle
from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--path', type=str, default='../data', help='file location')
    parser.add_argument('--source_dom', type=str, help='source domain')
    parser.add_argument('--tgt_dom', type=str, help='target domain')
    parser.add_argument('--batch_size', type=int,  help='batch size, should be even')
    parser.add_argument('--weight_decay', type=float,  help='weight decay rate')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--dropout', type=float,
                        help='the amount of dropout for the features')
    # action=store_true causes the parameter to be true when passed and false when missing
    parser.add_argument('--batch_norm', action='store_true',
                        help='use batch normalization in the feature extractor')
    parser.add_argument('--normalize', type=str, help='image normalization')
    parser.add_argument('--lr_disc', type=float, default=None, help='the learning rate of the discriminator')
    parser.add_argument('--gaussian_feature_perturbation_factor', type=float,
                        help='the amount of random noise added to the features before being passed to the disc.,'
                             'useful if the discriminator "learns too fast"')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.lr_disc is None:
        args.lr_disc = args.lr

    print(f"Passed parameters are {args}")
    if args.batch_norm and args.dropout > 0.0:
        print('Warning: The joint use of batch normalization and dropout is advised against.')

    assert args.batch_size % 2 == 0, 'Provided batch size is not even.'

    if args.path is None:
        data_path = pathlib.Path(
            '../data')
    else:
        data_path = pathlib.Path(args.path)

    source_dom = load_dataset_from_pickle(data_path / args.source_dom)
    prepare_data(source_dom, data_name=args.source_dom)
    tgt_dom = load_dataset_from_pickle(data_path / args.tgt_dom)
    prepare_data(tgt_dom, data_name=args.tgt_dom)

    if args.normalize is not None:
        if args.normalize == 'stdmean':
            print('Using stdmean normalization.')
            stdmean_normalize_dataset(source_dom)
            stdmean_normalize_dataset(tgt_dom)
        elif args.normalize == 'zeroone':
            print('Using zeroone normalization.')
            zeroone_normalize_dataset(source_dom)
            zeroone_normalize_dataset(tgt_dom)
        else:
            raise ValueError(f'Normalization parameter {args.normalize} not valid.')

    else:
        print('No normalization used.')

    tgt_dom_loader = PtLoader(tgt_dom, batch_size=int(args.batch_size / 2), drop_last=True)
    source_dom_loader = PtLoader(source_dom, batch_size=int(args.batch_size / 2), drop_last=True)

    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Warning: Option --cuda passed but no GPU found. Using CPU instead.")
            device = torch.device('cpu')
    else:
        device = torch.device("cpu")

    print(f'Training on {device}.')

    feat_ext = FeatureExtractor(dropout=args.dropout, feat_bn=args.batch_norm)
    discriminator = DomainClassifier(in_size=6272, n_domains=2)
    task_cls = TaskClassifier(in_size=6272)

    if args.wandb:
        # init and set up wandb
        wandb.init(project='2domadv')
        config = wandb.config
        for k in args.__dict__.keys():
            config[k] = args.__dict__[k]

        # watch models in wandb to track grads and params
        wandb.watch(feat_ext)
        wandb.watch(task_cls)
        wandb.watch(discriminator)

    feat_ext.to(device)
    discriminator.to(device)
    task_cls.to(device)

    criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()

    opt_feat = optim.SGD(feat_ext.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt_disc = optim.SGD(discriminator.parameters(), lr=args.lr_disc, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    opt_task = optim.SGD(task_cls.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    dom2loader_iter = cycle(tgt_dom_loader.train_loader)

    for epoch in range(args.epochs):

        train_data = {}
        train_data['tl'] = 0
        train_data['dl'] = 0
        train_data['task_acc'] = 0
        task_total = 0
        task_correct = 0
        dom_total = 0
        dom_correct = 0

        prog = tqdm(iter(source_dom_loader.train_loader), desc=f'Epoch {epoch}: ',
                    total=len(source_dom_loader.train_loader), postfix=train_data)

        # iterate over the source domain
        for d in prog:

            # get the inputs, keep labels
            inputs, labels = d
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            # get adversarial inputs, throw labels away
            adv_inputs, _ = next(dom2loader_iter)
            adv_inputs = adv_inputs.to(device).float()

            feat_ext.zero_grad()
            discriminator.zero_grad()
            task_cls.zero_grad()

            source_mask = torch.tensor([0] * inputs.shape[0]).long().to(device)
            target_mask = torch.tensor([1] * adv_inputs.shape[0]).long().to(device)

            assert inputs.shape == adv_inputs.shape
            total_inputs = torch.cat([inputs, adv_inputs])
            domain_labels = torch.cat([source_mask, target_mask])

            if args.gaussian_feature_perturbation_factor is not None and args.gaussian_feature_perturbation_factor > 0.0:
                total_feat = add_noise(inputs, args.gaussian_feature_perturbation_factor)

            source_feat = feat_ext(inputs)
            source_pred = task_cls(source_feat)
            tgt_loss = criterion(source_pred, labels.long())
            train_data['tl'] += tgt_loss.item() / len(source_dom_loader.train_loader)

            _, predicted = source_pred.max(1)
            task_total += labels.size(0)
            task_correct += predicted.eq(labels).sum().item()
            train_data['task_acc'] = task_correct / task_total
            tgt_loss.backward()
            opt_feat.step()
            opt_task.step()

            feat_ext.zero_grad()
            discriminator.zero_grad()
            task_cls.zero_grad()

            total_feat = feat_ext(total_inputs)

            if args.gaussian_feature_perturbation_factor is not None and args.gaussian_feature_perturbation_factor > 0.0:
                total_feat = add_noise(total_feat, args.gaussian_feature_perturbation_factor)

            domain_pred = discriminator(total_feat).squeeze()
            _, dom_predicted = domain_pred.max(1)
            dom_total += domain_labels.size(0)
            dom_correct += dom_predicted.eq(domain_labels).sum().item()
            train_data['dom_acc'] = dom_correct / dom_total
            dom_loss = adv_criterion(domain_pred, domain_labels.long())
            train_data['dl'] += dom_loss.item() / len(source_dom_loader.train_loader)
            dom_loss.backward()
            opt_feat.step()
            opt_disc.step()

            feat_ext.zero_grad()
            discriminator.zero_grad()
            task_cls.zero_grad()

            # show train data and update prog
            prog.set_postfix(train_data)

            if args.wandb:
                wandb.log(train_data)

        print(f'Epoch {epoch}, Data: {train_data}')

        forward_model = TaskPipeline(feature_extractor=feat_ext, task_classifier=task_cls)
        test_model(forward_model, source_dom_loader.test_loader, device=device, data_name=args.source_dom)
        test_model(forward_model, tgt_dom_loader.test_loader, device=device, data_name=args.tgt_dom)
