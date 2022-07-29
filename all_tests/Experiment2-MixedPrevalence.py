import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from core_functions import HelperFuncs
import seaborn as sns
import glob
import re

# Initialize a mixture of hw and comp with tags at index 0
# Let only the comp orgs reorganize their genes
# Calculate fitness of the mixed population
# Selection based on these fitness values, but note that competency needs to be phenotypic only.
# Get new population. Note down the number of hw and comp organisms. 
# Plot a graph of hw_f, comp_f vs #gen

class MixedFit():
    def __init__(self, conf, cfs):
        self.config = conf
        self.cfs = cfs
        self.hw_count = 0
        self.comp_count = 0

    def initialize(self):
        self.hw_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_hw'], self.config['SIZE'])) # hw = -2

        self.comp_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.config['N_comp'], self.config['SIZE'])) # comp = -1

        self.hw_organisms = np.array([np.insert(i, 0, -2, axis=0) for i in self.hw_organisms]) 
        self.comp_organisms = np.array([np.insert(i, 0, -1, axis=0) for i in self.comp_organisms])
        self.total_orgs = np.append(self.hw_organisms, self.comp_organisms, axis = 0)

    def count_orgs(self):
        t = [i for i in self.total_orgs if i[0] == -1]
        self.comp_count = len(t) / len(self.total_orgs)
        self.hw_count = 1 - self.comp_count 
        #self.hw_count = len(self.total_orgs) - self.comp_count 

    def calc_fitness_separately(self):
        hw_population  = [cfs.fitness(i[1:])[0] for i in self.total_orgs if i[0]==-2]
        comp_population = [cfs.fitness(i[1:])[0] for i in self.total_orgs if i[0]==-1]
        if len(hw_population):
            m_hw = max(hw_population)
        else:
            m_hw =0

        if len(comp_population):
            m_cp = max(comp_population)
        else:
            m_cp = 0

        return (m_hw, m_cp)


    def run_ga(self):

        self.initialize()

        self.hw_comp_fitness = [] 
        generation = 1

        self.hw_comp_fitness.append(self.calc_fitness_separately())

        self.counts = []
        self.count_orgs()
        print('Generation: {} | %HW: {} | %COMP: {}| Total_orgs:{}'.format(generation, self.hw_count, self.comp_count, len(self.total_orgs)))
        self.counts.append((self.hw_count, self.comp_count))

        while generation<N_generens:#(self.comp_count/len(self.total_orgs)) < 1.0 and (self.hw_count/len(self.total_orgs))< 1.0:  
        
            fittest_organisms, _ = self.cfs.combined_selection(self.total_orgs)
            new_population = self.cfs.crossover_mutation(fittest_organisms)
            mutated_population = self.cfs.mutation_flip(new_population)

            self.total_orgs = mutated_population.copy()               

            generation +=1

            self.hw_comp_fitness.append(self.calc_fitness_separately())

            self.count_orgs()
            self.counts.append((self.hw_count, self.comp_count)) 
            print('Generation: {} | %HW: {} | %COMP: {} | Total:{} ; Length: {}'.format(generation, self.hw_count, self.comp_count, len(self.total_orgs), self.total_orgs.shape))

        return self.hw_comp_fitness, self.counts


    def plot_counts(self, x_lab='Generations', y_lab='%'): 
        hw_count = [hw for hw,_ in self.counts]
        comp_count = [comp for _, comp in self.counts]
        # plt.rcParams["figure.figsize"] = (20,10)
        plt.plot(list(range(len(self.counts))), hw_count, label = 'HW count')
        plt.plot(list(range(len(self.counts))),comp_count, label = 'Comp count')
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.title('Mixed population counts per Generation')
        plt.legend()
        plt.savefig('counts')

    
def plot_both_fitness(self, x_lab='Generations', y_lab='Max Fitness'): 
    hw_fi = [hw for hw,_ in self.hw_comp_fitness]
    comp_fi = [comp for _, comp in self.hw_comp_fitness]

    hw_mean = np.mean(hw_fi)
    hw_std = np.std(hw_fi)

    comp_mean = np.mean(comp_fi)
    comp_std = np.std(comp_fi)
    t = list(range(1, len(self.hw_comp_fitness)+1))

    # plt.rcParams["figure.figsize"] = (20,10)
    plt.plot(t, hw_fi, label = 'HW fitness')
    plt.plot(t ,comp_fi, label = 'Comp fitness')
    plt.fill_between(t, hw_mean+hw_std, hw_mean-hw_std, alpha=0.3)
    plt.fill_between(t, comp_mean+comp_std, comp_mean-comp_std, alpha=0.3)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title('Fitness vs Generation (Mixed Population)')
    plt.legend()
    plt.savefig('fitness_mixed')

sns.set_theme(style = "darkgrid")
plt.rcParams["figure.figsize"] = (7.1, 9)
sns.set_palette(sns.color_palette())


if __name__ == "__main__":

    conf = {
    'LOW' :1,
    'HIGH':51,
    'N_hw' : 180,
    'N_comp':20,
    'N_organisms': 200,
    'SIZE' : 50,
    'Mutation_Probability' : 0.6,
    #'LIMIT': 50,
    #'MAX_ITER' : 100,
    #'MAX_FIELD' : 50,
    #'MODE': 'normal',
    #'REORG_TYPE': 2,
    'ExpFitness': True,
    'BubbleLimit': 35,
    'viewfield': 1,
    'Loops': 1,
    }

    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(10,13))
    N_runs =20
    N_generens = 20
    # results_hw = np.zeros((N_runs, N_generens))
    # results_comp = np.zeros((N_runs, N_generens))
    # preva_hw = np.zeros((N_runs, N_generens))
    # preva_comp = np.zeros((N_runs, N_generens))

    # for r in range(N_runs):
    #     cfs = HelperFuncs(conf)
    #     run1 = MixedFit(conf, cfs)
    #     te_buff, all_cnts = run1.run_ga()
    #     results_hw[r] = [hw for hw,_ in te_buff]
    #     results_comp[r] = [cmp for _, cmp in te_buff] 
    #     preva_hw[r] = [hw for hw, _ in all_cnts]
    #     preva_comp[r] = [comp for _, comp in all_cnts]

    # mr_hw = np.mean(preva_hw, axis=0)
    # vr_hw = np.std(preva_hw, axis =0)/np.sqrt(N_runs)

    # mr_cmp = np.mean(preva_comp, axis = 0)
    # vr_cmp = np.std(preva_comp, axis = 0)/np.sqrt(N_runs)

    # np.save('./exp2/hw_runs_preva{}_{}_{}'.format(conf['N_hw'], conf['N_comp'], conf['BubbleLimit']), preva_hw)
    # np.save('./exp2/comp_runs_preva{}_{}_{}'.format(conf['N_hw'], conf['N_comp'], conf['BubbleLimit']), preva_comp)

    # mr_hw = np.mean(results_hw, axis=0)
    # vr_hw = np.std(results_hw, axis =0)/np.sqrt(N_runs)
    # mr_cmp = np.mean(results_comp, axis = 0)
    # vr_cmp = np.std(results_comp, axis = 0)/np.sqrt(N_runs)



    file_list_comp = ['exp2/comp_runs_preva20_180_20.npy', 'exp2/comp_runs_preva100_100_20.npy', 'exp2/comp_runs_preva180_20_20.npy', 
    'exp2/comp_runs_preva20_180_25.npy', 'exp2/comp_runs_preva100_100_25.npy', 'exp2/comp_runs_preva180_20_25.npy', 
    'exp2/comp_runs_preva20_180_30.npy', 'exp2/comp_runs_preva100_100_30.npy', 'exp2/comp_runs_preva180_20_30.npy', 
    'exp2/comp_runs_preva20_180_35.npy', 'exp2/comp_runs_preva100_100_35.npy', 'exp2/comp_runs_preva180_20_35.npy']

    file_list_hw = ['exp2/hw_runs_preva20_180_20.npy', 'exp2/hw_runs_preva100_100_20.npy', 'exp2/hw_runs_preva180_20_20.npy', 
    'exp2/hw_runs_preva20_180_25.npy', 'exp2/hw_runs_preva100_100_25.npy', 'exp2/hw_runs_preva180_20_25.npy', 
    'exp2/hw_runs_preva20_180_30.npy', 'exp2/hw_runs_preva100_100_30.npy', 'exp2/hw_runs_preva180_20_30.npy', 
    'exp2/hw_runs_preva20_180_35.npy', 'exp2/hw_runs_preva100_100_35.npy', 'exp2/hw_runs_preva180_20_35.npy']

    row, col = 0,0
    bubbs = [20, 25, 30, 35]

    pr_hw = [20, 100, 180]
    pr_cmp = [180, 100, 20]
    for c, h in zip(file_list_comp, file_list_hw):

        preva_comp = np.load(c) 
        preva_hw = np.load(h)

        mr_hw = np.mean(preva_hw, axis=0)
        vr_hw = np.std(preva_hw, axis =0)/np.sqrt(N_runs)

        mr_cmp = np.mean(preva_comp, axis = 0)
        vr_cmp = np.std(preva_comp, axis = 0)/np.sqrt(N_runs) 

        htrace, =axs[row,col].plot(range(1, N_generens+1), mr_hw)#, label='Hardwired')
        axs[row, col].fill_between(range(1,N_generens+1), mr_hw-1*vr_hw, mr_hw+1*vr_hw, alpha =0.2) 

        ctrace, = axs[row,col].plot(range(1, N_generens+1), mr_cmp)#, label='Competent')
        axs[row, col].fill_between(range(1,N_generens+1), mr_cmp-1*vr_cmp, mr_cmp+1*vr_cmp, alpha =0.2) 
        axs[row, col].legend()
        if col==0:
            axs[row, col].set_ylabel('{} Swaps'.format(bubbs[row]))

        if row==0: 
            axs[row, col].set_title('{} HW - {} Competent'.format(pr_hw[col], pr_cmp[col]))

        col+=1

        if col==3:
            col = 0
            row += 1

    preva_hw = np.load('exp2/hw_runs_preva100_100_20.npy')
    preva_comp = np.load('exp2/comp_runs_preva100_100_20.npy')
    mr_hw = np.mean(preva_hw, axis=0)
    vr_hw = np.std(preva_hw, axis =0)/np.sqrt(N_runs)

    mr_cmp = np.mean(preva_comp, axis = 0)
    vr_cmp = np.std(preva_comp, axis = 0)/np.sqrt(N_runs)

    # plt.rcParams["figure.figsize"] = (4,3)
    # plt.plot(range(1, N_generens+1), mr_hw, label='Hardwired')
    # plt.fill_between(range(1,N_generens+1), mr_hw-1*vr_hw, mr_hw+1*vr_hw, alpha =0.2) 

    # plt.plot(range(1, N_generens+1), mr_cmp, label='Competent, {} swaps'.format(conf['BubbleLimit']))
    # plt.fill_between(range(1,N_generens+1), mr_cmp-1*vr_cmp, mr_cmp+1*vr_cmp, alpha =0.2)
    # plt.xlabel('Generation')
    # plt.ylabel("Percentage of Individuals (%)")
    # plt.legend(loc='center right', prop = {'size': 8.5})
    # leg1 = axs[0,1].legend([ctrace, htrace], ['Competent', 'Hardwired'], loc = 'upper center', bbox_to_anchor = (0.0, 1.4), fancybox =True, shadow = True)
    # fig.gca().add_artist(leg1)
    fig.subplots_adjust(top=0.9)
    axs[0,1].legend([ctrace, htrace], ['Competent', 'Hardwired'], loc = 'upper center', bbox_to_anchor = (0.50, 1.35), fancybox =True, shadow = True)

    fig.supxlabel('Generation')
    fig.supylabel('Percentage of Individuals (%)')
    plt.tight_layout()
    fig.savefig('./final_figures/Exp2:A', dpi=300)
    # plt.show()

    plt.show()