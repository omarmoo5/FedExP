# Initiate the NN

from util_data import *
from util_general import *
from util_models import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_clients', type=int, required=True)
parser.add_argument('--num_participating_clients', type=int, required=True)
parser.add_argument('--num_rounds', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)

args_required = parser.parse_args()

seed = args_required.seed
dataset = args_required.dataset
algorithm = args_required.algorithm
model = args_required.model
num_clients = args_required.num_clients
num_participating_clients = args_required.num_participating_clients
num_rounds = args_required.num_rounds
alpha = args_required.alpha

print_every_test = 5
print_every_train = 5

filename = "_".join(["results", str(seed), algorithm, dataset, model,
                     str(num_clients), str(num_participating_clients),
                     str(num_rounds), str(alpha)])
filename_txt = filename + ".txt"

n_c = 100 if dataset == 'CIFAR100' else 10

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

dataset_train, dataset_test_global = get_dataset(dataset, num_clients, n_c, alpha, True)

dict_results = {}  # dictionary to store results for all algorithms

# Default training parameters for all algorithms

args = {"bs": 50,  # batch size
        "cp": 20,  # number of local steps
        "device": 'cpu',
        "rounds": num_rounds,
        "num_clients": num_clients,
        "num_participating_clients": num_participating_clients
        }

net_glob_org = get_model(model, n_c).to(args['device'])

algs = [algorithm]

decay = 0.998
max_norm = 10
use_gradient_clipping = True
weight_decay = 0.0001

if dataset in ['CIFAR10', 'CIFAR100']:
    eta_l_fedavg = 0.01
    eta_l_fedexp = 0.01
    eta_g_fedavg = 1
    epsilon_fedexp = 0.001
else:
    raise ValueError('Dataset not supported')

eta_l_algs = {'fedavg': eta_l_fedavg,
              'fedexp': eta_l_fedexp,
              }

eta_g_algs = {'fedavg': eta_g_fedavg,
              'fedexp': 'adaptive',
              }

epsilon_algs = {'fedavg': 0,
                'fedexp': epsilon_fedexp,
                }

n = len(dataset_train)
print("No. of clients", n)

p = np.zeros(n)
for i in range(n):
    p[i] = len(dataset_train[i])
p /= np.sum(p)

for alg in algs:
    dict_results[alg] = {}
    filename_model_alg = "_".join([alg, filename]) + ".pt"
    d = parameters_to_vector(net_glob_org.parameters()).numel()
    net_glob = copy.deepcopy(net_glob_org)
    net_glob.train()
    w_glob = net_glob.state_dict()

    train_loss_algo_tmp = []
    train_acc_algo_tmp = []
    test_loss_algo_tmp = []
    test_acc_algo_tmp = []
    eta_g_tmp = []

    w_vec_estimate = torch.zeros(d).to(args['device'])
    local_lr = eta_l_algs[alg]
    global_lr = eta_g_algs[alg]
    epsilon = epsilon_algs[alg]

    for t in tqdm(range(0, args['rounds'] + 1), desc=f"Algorithm {alg}"):
        print("Algo ", alg, " Round No. ", t)
        local_lr *= decay
        epsilon *= decay ** 2
        args_hyperparameters = {'eta_l': local_lr,
                                'decay': decay,
                                'weight_decay': weight_decay,
                                'eta_g': global_lr,
                                'use_gradient_clipping': use_gradient_clipping,
                                'max_norm': max_norm,
                                'epsilon': epsilon,
                                'use_augmentation': True
                                }
        p_sum, grad_norm_sum, grad_avg = 0, 0, torch.zeros(d).to(args['device'])
        # Client Training
        S = args['num_participating_clients']
        clients_ids = np.random.choice(n, S, replace=False)
        for i in tqdm(clients_ids, desc=f"Local Training of round {t}"):
            grad = get_grad(copy.deepcopy(net_glob),
                            args, args_hyperparameters,
                            dataset_train[i], alg)
            p_sum += p[i]
            grad_avg += p[i] * grad
            grad_norm_sum += p[i] * torch.linalg.norm(grad) ** 2

        with torch.no_grad():
            grad_avg /= p_sum
            grad_avg_norm = torch.linalg.norm(grad_avg) ** 2
            grad_norm_avg = grad_norm_sum / p_sum

            eta_g = args_hyperparameters['eta_g']
            if eta_g == 'adaptive':
                eta_g = max(1, (0.5 * grad_norm_avg / (grad_avg_norm + S * epsilon)).cpu())
            eta_g_tmp.append(eta_g)

            w_vec_prev = w_vec_estimate
            w_vec_estimate = parameters_to_vector(net_glob.parameters()) + eta_g * grad_avg

            w_vec_avg = w_vec_estimate if t == 0 else (w_vec_estimate + w_vec_prev) / 2
            vector_to_parameters(w_vec_estimate, net_glob.parameters())

        net_eval = copy.deepcopy(net_glob)
        if alg == 'fedexp':
            vector_to_parameters(w_vec_avg, net_eval.parameters())

        if t % print_every_test == 0:
            print("Testing 1")
            if t % print_every_train == 0:
                print("Testing 2")
                sum_loss_train = 0
                sum_acc_train = 0
                for i in tqdm(range(n), desc=f"Testing on training data"):
                    test_acc_i, test_loss_i = test_img(net_eval, dataset_train[i], args)
                    sum_loss_train += test_loss_i
                    sum_acc_train += test_acc_i
                sum_loss_train /= n
                sum_acc_train /= n
                print("Training Loss ", sum_loss_train, "Training Accuracy ", sum_acc_train)
                train_loss_algo_tmp.append(sum_loss_train)
                train_acc_algo_tmp.append(sum_acc_train)
            sum_loss_test = 0
            test_acc_i, test_loss_i = test_img(net_eval, dataset_test_global, args)
            print("Test Loss", test_loss_i, "Test Accuracy ", test_acc_i)
            test_loss_algo_tmp.append(test_loss_i)
            test_acc_algo_tmp.append(test_acc_i)
            dict_results[alg][alg + "_training_loss"] = train_loss_algo_tmp
            dict_results[alg][alg + "_training_accuracy"] = train_acc_algo_tmp
            dict_results[alg][alg + "_test_loss"] = test_loss_algo_tmp
            dict_results[alg][alg + "_testing_accuracy"] = test_acc_algo_tmp
            dict_results[alg][alg + "_global_learning_rate"] = eta_g_tmp
            torch.save(net_glob, filename_model_alg)
            with open(filename_txt, 'w') as f:
                for i in dict_results.keys():
                    for key, value in dict_results[i].items():
                        f.write(key + " ")
                        f.write(str(value))
                        f.write("\n")
