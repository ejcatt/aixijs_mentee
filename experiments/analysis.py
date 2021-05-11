import numpy as np
import matplotlib.pyplot as plt
import json

import matplotlib as mpl
label_size = 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

agent_labels = {'BayesAgent':(r'AI$\xi$','red'),
          'MC-AIXI':('MC-AIXI','red'),
          'MC-AIMU':('MC-AIMU','blue'),
          'MDL Agent':('MDL','blue'),
          'MC-AIXI-Dirichlet':('MC-AIXI-Dirichlet','blue'),
          'Knowledge-seeking agent':('Kullback-Leibler','blue'),
          'KullbackLeiblerKSA':('Kullback-Leibler','blue'),
          'ShannonKSA':('Shannon','green'),
          'SquareKSA':('Square','red'),
          'Shannon KSA':('Shannon','orange'),
          'Square KSA':('Square','red'),
          'ThompsonAgent':('Thompson Sampling','blue'),
          'Thompson Sampling':('Thompson Sampling','blue'),
          'QLearn':('Q-Learning','black'),
          'Q-Learning':('Q-Learning','black'),
          'KSA-Dirichlet': ('Kullback-Leibler','blue'),
          'Entropy-seeking agent': ('Shannon','orange'),
          'Square KSA-Dirichlet': ('Square','red'),
          'BayesExp': ('BayesExp','red'),
          'Inq': ('Inq','orange'),
        'Mentor': ('Mentor','black'),
         'Mentee': ('Mentee','green')      }

def plot_results(directory,
                 filename='results-1',
                 objective=None,
                 outfile=None,
                 show_optimal=False,
                 show_variance=True,
                 show_maxmin=False,
                 show_variance_mean=False,
                 leg_loc = 'lower left'):

    # some cruft to add default labels
    if not objective:
        if 'ksa' in directory:
            objective = 'explored'
        else:
            objective = 'rewards'
    if objective == 'rewards':
        y_axis = 'Average Reward'
    elif objective == 'explored':
        y_axis = 'Exploration (%)'

    file = open(directory + '/' + filename + '.json')
    data = json.load(file)
    file.close()

    fig = plt.figure(figsize=(12,8),dpi=200)
    # iterate over configs
    for i,k in enumerate(data):
        try:
            d = data[k]
        except KeyError:
            continue
        cycles = 8000 #d[0]['cycles']
        runs = len(d)

        A = np.zeros((cycles,runs))
        for j in range(runs):
            A[:,j] = np.array(d[j][objective][:cycles])
        mu = np.mean(A,1)
        sigma = np.std(A,1)
        sigma_mean = sigma/np.sqrt(runs)
        a = np.max(np.vstack((mu-sigma,np.min(A,1))),0)
        b = np.min(np.vstack((mu+sigma,np.array(cycles*[100]))),0)
        a_mean = np.max(np.vstack((mu-sigma_mean,np.min(A,1))),0)
        b_mean = np.min(np.vstack((mu+sigma_mean,np.array(cycles*[100]))),0)

        if k in agent_labels:
            lab = agent_labels[k][0]

        color = agent_labels[k][1]
        alpha = 0.1

        if show_variance:
            plt.plot(a,color=color,alpha=alpha)
            plt.plot(b,color=color,alpha=alpha)
            plt.fill_between(np.arange(cycles),a,b,alpha=alpha,color=color)

        if show_variance_mean:
            plt.plot(a_mean,color=color,alpha=alpha)
            plt.plot(b_mean,color=color,alpha=alpha)
            plt.fill_between(np.arange(cycles),a_mean,b_mean,alpha=alpha,color=color)

        if show_maxmin:
            plt.plot(np.max(A,axis=1),color=color,linestyle='-.')
            plt.plot(np.min(A,axis=1),color=color,linestyle='-.')

        plt.xscale('log')
        plt.plot(mu,label=lab,color=color,lw=3)


    if objective=='rewards' and show_optimal:
        # NOTE: hardcoded for optimal policy in one gridworld
        xs = np.array(range(cycles))
        ys = np.zeros(cycles)
        ys[:11] = -1.
        ys[11:] = 75.
        ys = np.cumsum(ys)

        ys[1:] /= xs[1:]
        plt.plot(xs,ys,'k--',lw=3,label='Optimal')

    plt.xlabel('Cycles',fontsize=20)
    plt.ylabel(y_axis,fontsize=20)
    plt.legend(fontsize=15,loc=leg_loc)
    plt.margins(0.01,0)
    #plt.ylim([-1,100])

    if outfile:
        plt.savefig(directory + '/' + outfile + '.png', bbox_inches='tight')
        plt.close()

plot_results('new-experiments','results-1',outfile='test-8000-eta0.1-funky-trap20',show_optimal=False,show_variance=False,show_variance_mean=True,leg_loc='upper left')
# plot_results('new-experiments','results-ao',outfile='ao-trap20',show_optimal=False,show_variance=False,show_variance_mean=True,leg_loc='upper right')
