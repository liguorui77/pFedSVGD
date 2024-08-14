import torch
import torch.nn.functional as F
import numpy as np
import random
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from Library.bayesian_neural_networks_classification import evaluation, compute_gradient
from Library.general_functions import svgd_kernel, pairwise_distances, kde
from Library.model_utils import read_data, read_user_data
from Library.model.ResNet18 import ResNet18



def agent_dsvgd(id, i, model, M, nb_svgd, nb_svgd_2, y_train, X_train, global_particles, personal_particle, likelihood_particle, batchsize, device):
    sum_squared_grad = torch.zeros([M, num_vars], device= device)  # sum of gradient squared to use in Ada grad learning rate for first SVGD
    sum_squared_grad_2 = torch.zeros([M, num_vars], device= device)  # sum of gradient squared to use in Ada grad learning rate for second svgd
    
    if i == 0:
        # It's the first iteration for each agent when i = 0
        particles = (global_particles.detach().clone()).to(device)
        particles.requires_grad = False
        particles_2 = (likelihood_particle.detach().clone()).to(device)
        particles_2.requires_grad = False
    else:
        particles = global_particles.detach().clone().to(device)
        particles.requires_grad = True

        N0 = X_train.shape[0]  
        grad_theta = torch.zeros([M, num_vars],device = device)  # gradient

        ''' First SVGD loop'''
        for t in range(0, nb_svgd):
            kxy, dxkxy = svgd_kernel(particles.detach().clone(), h=-1)
            
            distance_M_j = pairwise_distances(M, particles.transpose(0, 1), likelihood_particle.to(device).transpose(0, 1))
            t_prev = kde(M, num_vars, my_lambda, distance_M_j)

            distance_M_i_1 = pairwise_distances(M, particles.transpose(0, 1), global_particles.to(device).transpose(0, 1))
            qi_1 = kde(M, num_vars, my_lambda, distance_M_i_1)

            # compute target distribution
            sv_target = torch.log(qi_1 + 10**(-10)) - torch.log(t_prev + 10**(-10))
            
            ''' Compute grad_theta '''
            # sub-sampling
            batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
            zeta = particles.detach().clone()
            for m in range(M):
                grad_theta[m, :] = compute_gradient(X_train[batch, :], y_train[batch], zeta[m, :], model, num_vars, loglambda, N=N0)

            sv_target.backward(torch.ones(M, dtype=torch.double,device= device))
            grad_sv_target =(1/alpha) *grad_theta.clone() + particles.grad.clone()
            particles.grad.zero_()

            # compute delta_theta used to update all particles
            if M != 1:
                delta_theta = (1/M) * (torch.mm(kxy.clone(), grad_sv_target) + dxkxy.clone())
            else:
                delta_theta = grad_sv_target

            if t == 0:
                sum_squared_grad = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad = betta * sum_squared_grad + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad))
            # update global particles
            with torch.no_grad():
                particles = particles + epsilon_svgd * delta_theta.detach().clone()
            particles.requires_grad = True

        # End of first SVGD loop

        ''' Second SVGD loop'''
        particles.requires_grad = False
        
        torch.cuda.empty_cache()
        
        particles_2 = likelihood_particle.detach().clone().to(device)
        particles_2.requires_grad = True

        for t in range(nb_svgd_2):
            kxy, dxkxy = svgd_kernel(particles_2.detach().clone(), h=-1)

            distance_M_t = pairwise_distances(M, particles_2.transpose(0, 1), likelihood_particle.to(device).transpose(0, 1))
            t_prev = kde(M, num_vars, my_lambda, distance_M_t)
            log_t_prev = torch.log(t_prev + 10**(-10))

            distance_M_i = pairwise_distances(M, particles_2.transpose(0, 1), particles.transpose(0, 1))
            qi = kde(M, num_vars, my_lambda, distance_M_i)
            log_qi = torch.log(qi + 10**(-10))

            distance_M_i_1 = pairwise_distances(M, particles_2.transpose(0, 1), global_particles.to(device).transpose(0, 1)) # global_particles
            qi_1 = kde(M, num_vars, my_lambda, distance_M_i_1)
            log_qi_1 = - torch.log(qi_1 + 10**(-10))

            # compute target distribution
            if t_prev.requires_grad:
                log_t_prev.backward(torch.ones(M, dtype=torch.double,device=device))
                log_qi.backward(torch.ones(M, dtype=torch.double,device=device))
                log_qi_1.backward(torch.ones(M, dtype=torch.double,device=device))
                grad_sv_target = particles_2.grad.clone()
                particles_2.grad.zero_()
            else:
                print('Yooooo')
                grad_sv_target = 0
            # compute delta_theta used to update all particles
            delta_theta = (1/M) * (torch.mm(kxy.clone(), grad_sv_target) + dxkxy.clone())

            if t == 0:
                sum_squared_grad_2 = torch.pow(delta_theta.detach().clone(), 2)
            else:
                sum_squared_grad_2 = betta * sum_squared_grad_2 + (1-betta) * torch.pow(delta_theta.detach().clone(), 2)

            epsilon_svgd = alpha_ada / (epsilon_ada + torch.sqrt(sum_squared_grad_2))
            # update local particles
            with torch.no_grad():
                particles_2 = particles_2 + epsilon_svgd * delta_theta.detach().clone()
            particles_2.requires_grad = True
        
        particles_2.requires_grad = False
    return particles.to(torch.device('cpu')), particles_2.to(torch.device('cpu'))


def server(nb_devices, particles, model , M, nb_svgd, nb_svgd_2, nb_global, y, X, y_test, X_test, batchsize, device):
    """
    Function that simulates the central server and schedules agents
    """

    # initialize local particles at each device
    personal_particles_pre = particles.repeat(nb_devices, 1, 1).to(torch.device('cpu'))
    personal_particles = particles.repeat(nb_devices, 1, 1).to(torch.device('cpu'))
    likelihood_particles = particles.repeat(nb_devices, 1, 1).to(torch.device('cpu'))
    _acc_g = np.zeros(nb_global)
    _llh_g = np.zeros(nb_global)
    _acc_p = np.zeros(nb_global)
    _llh_p = np.zeros(nb_global)

    for i in range(0, nb_global):
        # schedule all agent
        
        for j in range(nb_devices):
            X_curr, y_curr = X[j], y[j]
            
            X_curr = X_curr.to(device)
            y_curr = y_curr.to(device)
            
            personal_particles[j], likelihood_particles[j] = agent_dsvgd(j+1, i, model,\
            M, nb_svgd, nb_svgd_2, y_curr, X_curr, particles, personal_particles[j], likelihood_particles[j], batchsize=batchsize, device=device)
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                acc_per, llh_per = evaluation(M, personal_particles[j].detach().clone(), model, number_labels, X_test[j], y_test[j] , device )
            
            if torch.isnan(personal_particles[j]).any().item() :
                personal_particles[j] = (1. + 10**(-3)) * personal_particles_pre[j].detach().clone()
                
                with torch.no_grad():
                    acc_per, llh_per = evaluation(M, personal_particles[j].detach().clone(), model, number_labels, X_test[j], y_test[j] , device )
            
            personal_particles_pre[j] = personal_particles[j].detach().clone()
            
            print(i,"   ",j,"   ", acc_per, "   ", llh_per)
            _acc_p[i] += acc_per
            _llh_p[i] += llh_per
            
            torch.cuda.empty_cache()
        
        _acc_p[i] /= nb_devices
        _llh_p[i] /= nb_devices
        
        
        particles_1 = torch.zeros([M, num_vars]).type(torch.float32).to(torch.device('cpu'))
        
        for k in range(nb_devices):
            particles_1 += 1.0 / nb_devices * personal_particles[k]
        
        particles = (1-betta_a) * particles.detach().clone() + (betta_a) * particles_1.detach().clone()

        for j in range(nb_devices):
            
            with torch.no_grad():
                acc_cur, llh_cur = evaluation(M, particles.detach().clone(), model, number_labels, X_test[j], y_test[j] , device )
            
            _acc_g[i] += (test_size[j] / test_size_sum ) * acc_cur
            _llh_g[i] += (test_size[j] / test_size_sum ) * llh_cur
        
        print(_acc_g[i], "   ", _llh_g[i])
    
    return _acc_g, _llh_g, _acc_p, _llh_p


if __name__ == '__main__':
    ''' Parameters'''
    torch.random.manual_seed(0)
    np.random.seed(0)
    alpha = 1.  # temperature parameter multiplying divergence of posterior with prior
    nb_svgd = 60  # number of iterations for first SVGD loop with target \tilde{p}  
    nb_svgd_2 = 60  # number of iterations for second SVGD loop with target t    
    nb_global = 100  # number of PVI iterations
    # my_lambda = 0.39894228  # bandwidth for kde_vectorized
    my_lambda = -1.
    number_labels = 10  # number of labels in MNIST
    alpha_ada = 10 ** (-3)  # constant rate used in AdaGrad
    epsilon_ada = 10 ** (-6)  # fudge factor for adagrad
    betta = 0.9  # for momentum update
    betta_a = 1.  #0.9 
    nb_exp = 1  # number of random trials
    nb_devices = 10  # number of agents
    batchsize = 128
    dataset='Cifar10'
    loglambda = 1  # log precision of weight prior
    M = 20  # number of particles
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(device).to(device)
    array_acc_g = np.zeros(nb_global)
    array_llh_g = np.zeros(nb_global)
    array_acc_p = np.zeros(nb_global)
    array_llh_p = np.zeros(nb_global)
    
    data = read_data(dataset)
    train_imgs = []
    test_imgs = []
    train_labels = []
    test_labels = []
    test_size = np.zeros(nb_devices)
    
    i=0
    for user in data[0] :
        train_data=np.asarray(data[2][user]['x'],dtype=np.float64)
        train_labels_cur=np.asarray(data[2][user]['y']).astype(int)
        test_data=np.asarray(data[3][user]['x'],dtype=np.float64)
        test_labels_cur=np.asarray(data[3][user]['y']).astype(int)
        
        
        # shuffle the training and test set
        permutation = np.arange(train_data.shape[0])
        random.shuffle(permutation)
        index_train = permutation

        permutation = np.arange(test_data.shape[0])
        random.shuffle(permutation)
        index_test = permutation

        ''' build the training and testing data set '''
        X_train, y_train = train_data[index_train, :], train_labels_cur[index_train].flatten()
        X_test, y_test = test_data[index_test, : ], test_labels_cur[index_test].flatten()
        
        X_train = torch.Tensor(X_train).type(torch.float32).to(torch.device('cpu'))
        y_train = torch.Tensor(y_train).type(torch.int64).to(torch.device('cpu'))
        X_test = torch.Tensor(X_test).type(torch.float32).to(torch.device('cpu'))
        y_test = torch.Tensor(y_test).type(torch.int64).to(torch.device('cpu'))
        
        train_imgs.append(X_train)
        train_labels.append(y_train)
        test_imgs.append(X_test)
        test_labels.append(y_test)
        test_size[i]=len(y_test)
        i=i+1
    

    test_size_sum=np.sum(test_size)
    for exp in range(nb_exp):
        print('Trial ', exp + 1)

        num_vars = model.compute_total_parameters()

        ''' Initialize particles'''
        theta = torch.zeros([M, num_vars]).to(device)
        for i in range(M):
            model.reset_parameters()
            theta[i, :] = model.send_parameters()

        particles = torch.Tensor(theta).type(torch.float32).to(torch.device('cpu'))
        
        del theta
        
        ''' Run pFedSVGD server '''
        curr_acc_g, cur_llh_g, curr_acc_p, cur_llh_p = server(nb_devices, particles, model, M, nb_svgd, nb_svgd_2, nb_global, train_labels, train_imgs, test_labels, test_imgs, batchsize, device)
        array_acc_g += curr_acc_g
        array_llh_g += cur_llh_g
        array_acc_p += curr_acc_p
        array_llh_p += cur_llh_p

    
    print_log = open("/home/zjm/Experiments/pFedSVGD/output.txt",'w')

    print('global model accuracy of comm. rounds = ', repr(array_acc_g / nb_exp),file = print_log)
    print('global model max = ', repr(np.max(array_acc_g / nb_exp)),file = print_log)
    print('personal model accuracy of comm. rounds = ', repr(array_acc_p / nb_exp),file = print_log)
    print('personal model max = ', repr(np.max(array_acc_p / nb_exp)),file = print_log)
    
    print_log.close()

    print('global model accuracy of comm. rounds = ', repr(array_acc_g / nb_exp))
    print('personal model accuracy of comm. rounds = ', repr(array_acc_p / nb_exp))



