from tqdm import tqdm

from util_libs import *


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss


class LocalUpdate(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(), ])

    def train_and_sketch(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.lr,
                                    momentum=0,
                                    weight_decay=self.weight_decay)
        prev_net = copy.deepcopy(net)
        # Local Training
        for _ in tqdm(range(self.args['cp']), desc='Client Training'):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if self.use_data_augmentation:
                    images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
                optimizer.step()
        with torch.no_grad():
            vec_curr = parameters_to_vector(net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            params_delta_vec = vec_curr - vec_prev
            model_to_return = params_delta_vec
        return model_to_return


def get_grad(net_glob, args, args_hyperparameters, dataset, alg):
    if alg in ['fedexp', 'fedavg']:
        local = LocalUpdate(args, args_hyperparameters, dataset=dataset)
        grad = local.train_and_sketch(copy.deepcopy(net_glob))
        return grad
    else:
        raise ValueError('Algorithm not supported')
