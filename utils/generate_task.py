###### Generates the task ######

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# Function
def generate_task(trial_num = 1000, alpha = -1., v = .03, beta = .9, action_num = 2, state_num = 1, plot = True):
    # Switches
    tau           = np.zeros(trial_num)
    tau[:200]     = .03
    tau[200:300]  = .2
    tau[300:400]  = .03
    tau[400:500]  = .2
    tau[500:600]  = .03
    tau[600:700]  = .2
    tau[700:800]  = .03
    tau[800:900]  = .2
    tau[900:1000] = .03
    lim          = (factorial(action_num - 1)/(factorial(action_num - state_num) * factorial(state_num - 1))) + 3
    index        = 0
    B            = np.zeros(trial_num)
    for i in range(trial_num):
        if index > 0 and index < lim:
            index += 1
        else:
            B[i]  = np.random.rand() < tau[i]
            index = B[i] * 1

    B[0]          = 1;

    # Traps trials
    trap = np.random.rand(trial_num) < (1 - beta);
    
    # Hidden state sequence and Dirichlet prior parameters
    K = np.prod(np.arange(action_num+1)[-state_num:])
    Z = np.zeros(trial_num);
    if np.equal(alpha,-1) : alpha = K;
    Z_prior = alpha / K * np.ones(K);

    # Generate sequence of hidden variables
    for trial_idx in range(trial_num):
        if B[trial_idx]:
            # Pick new hidden variable and update prior
            Z[trial_idx]          = np.random.choice(K, p = Z_prior/sum(Z_prior))
            while (Z[trial_idx] == Z[trial_idx - 1]):
                Z[trial_idx]      = np.random.choice(K, p = Z_prior/sum(Z_prior))
            Z_prior[int(Z[trial_idx])] = Z_prior[int(Z[trial_idx])] + 1;
        else:
            # Do not change hidden variable
            Z[trial_idx] = Z[trial_idx - 1];
    
    # State - Stimulus sequence
    S = np.random.randint(state_num, size = trial_num);
    
    if plot:
        plt.figure();
        plt.subplot(2,1,1)
        plt.plot(tau, 'r', label='Switch Probability');
        plt.hold(True);
        plt.plot(B, 'g', label='Switch');
        plt.axis([0,trial_num,0,1]);
        plt.title('Switch and Switch Probability')
        plt.legend(fancybox=True);
        plt.subplot(2,1,2)
        plt.plot(Z, label='Task Set');
        plt.hold(True);
        #plt.plot(B, label='Switch');
        plt.axis([0,trial_num,0,K]);
        plt.title('Task Set')
        plt.legend(fancybox=True);
        
    # Generate dictionary structure
    return {'Z': Z, 'B': B, 'S': S, 'trap': trap, \
            'alpha':alpha, 'tau':tau, 'beta':beta,'v':v,  \
            'action_num':action_num,'state_num':state_num}
