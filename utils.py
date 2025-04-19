import time, torch
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import f1_score, roc_auc_score
import logging
from evaluate import evaluate

class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]


def train(train_loader, network, criterion, model_teacher, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    network.train()
    if model_teacher is not None:
        model_teacher.eval()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)

            # Compute output
            output = network(input)
            if model_teacher is not None:
                output_teacher = model_teacher(input)
                loss = criterion(output, output_teacher)
                losses.update(loss.item(), input.size(0))
            else:
                loss = criterion(output, target).mean()
                losses.update(loss.data.item(), input.size(0))

        # Measure accuracy and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))[0]
        #top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LR {lr:.5f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, lr=_get_learning_rate(optimizer)))
            
    scheduler.step()
    record_train_stats(rec, epoch, losses.avg, optimizer.state_dict()['param_groups'][0]['lr'])

def test(test_loader, network, criterion, epoch, args, rec):
    network.eval()
    network.no_grad = True

    all_preds = []
    all_labels = []
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.to(args.device)
            target = target.to(args.device).float()

            output = network(input)
            loss = criterion(output, target).mean()
            total_loss += loss.item() * input.size(0)
            count += input.size(0)

            probs = torch.sigmoid(output)
            all_preds.append(probs.cpu())
            all_labels.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    binary_preds = (all_preds > 0.5).astype(int)

    # Accuracy
    acc = (binary_preds == all_labels).mean(axis=1).mean()

    f1_macro = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, binary_preds, average='micro', zero_division=0)
    try:
        auc_macro = roc_auc_score(all_labels, all_preds, average='macro')
        auc_micro = roc_auc_score(all_labels, all_preds, average='micro')
    except ValueError:
        auc_macro, auc_micro = float('nan'), float('nan')

    avg_loss = total_loss / count

    logging.info(f"\n[Multi-label Evaluation]")
    logging.info(f"Loss       : {avg_loss:.4f}")
    logging.info(f"Accuracy   : {acc:.4f}")
    logging.info(f"F1 Macro   : {f1_macro:.4f}")
    logging.info(f"F1 Micro   : {f1_micro:.4f}")
    logging.info(f"AUC Macro  : {auc_macro:.4f}")
    logging.info(f"AUC Micro  : {auc_micro:.4f}")

    network.no_grad = False
    record_test_stats(rec, epoch, avg_loss, acc, f1_macro, f1_micro, auc_macro, auc_micro)

    save_recorder_to_json(rec, output_dir=args.output)

    # return acc
    return acc

def save_recorder_to_json(rec, output_dir="results", filename="recorder.json"):
    import json, os

    def convert(v):
        if isinstance(v, torch.Tensor):
            return v.tolist()
        return v

    rec_dict = {k: [convert(x) for x in v] for k, v in rec.__dict__.items()}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(rec_dict, f, indent=2)

    logging.info(f"Recorder saved to {output_path}")


# test for top-k
''' 
def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(args.device)
        input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def _get_learning_rate(optimizer):
    return max(param_group['lr'] for param_group in optimizer.param_groups)

def save_checkpoint(state, path, epoch, prec):
    logging.info("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    # Training
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.train_f1_macro = []
    rec.train_f1_micro = []
    rec.train_auc_macro = []
    rec.train_auc_micro = []
    rec.lr = []

    # Evaluation
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.test_f1_macro = []
    rec.test_f1_micro = []
    rec.test_auc_macro = []
    rec.test_auc_micro = []

    # Checkpoint records
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    #rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec



def record_test_stats(rec, step, loss, acc, f1_macro=None, f1_micro=None, auc_macro=None, auc_micro=None):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    rec.test_f1_macro.append(f1_macro)
    rec.test_f1_micro.append(f1_micro)
    rec.test_auc_macro.append(auc_macro)
    rec.test_auc_micro.append(auc_micro)
    return rec



def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
