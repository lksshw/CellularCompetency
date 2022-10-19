#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from core_functions import HelperFuncs

# seaborn settings
sns.set_theme(style = "darkgrid")
sns.set_palette(sns.color_palette())

class MixedFit():
    def __init__(self, conf, cfs, n_hw, n_comp):
        self.config = conf   # configuration file from the hyperparameters file
        self.cfs = cfs       # object of corefunctions
        self.hw_count = 0    # counter to keep track of hw individuals in the population
        self.comp_count = 0  # counter to keep track of competent individuals in the population
        self.n_hw = n_hw     # number of hw individual initialized
        self.n_comp = n_comp # number of competent individuals intialized

    def initialize(self):
        # Function to initialize the experiment. Involved creating individuals and assigning identifiers to indentify them.

        self.hw_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.n_hw, self.config['SIZE'])) # initialize n_hw number of hw individuals
        self.comp_organisms = np.random.randint(low=self.config['LOW'], high=self.config['HIGH'], size=(self.n_comp, self.config['SIZE'])) # initialize n_comp number of competent individuals

        self.hw_organisms = np.array([np.insert(i, 0, -2, axis=0) for i in self.hw_organisms])  # to indentify if an individual is hw or competent we assign id's to index 0 of each individual. HW index = -2, Competent index = -1
        self.comp_organisms = np.array([np.insert(i, 0, -1, axis=0) for i in self.comp_organisms])

        self.total_orgs = np.append(self.hw_organisms, self.comp_organisms, axis = 0) # combine the two sets of individuals

    def count_orgs(self):
        # Function to count the proportion of Hw and competent individuals in a population at any given time

        t = [i for i in self.total_orgs if i[0] == -1] # count number of competent based on identifier (i.e -1)
        self.comp_count = len(t) / len(self.total_orgs) # update competent count
        self.hw_count = 1 - self.comp_count  # update hw count

    def calc_fitness_separately(self):
        # Function to calculate the fitness of Hw and competent individuals separately.

        hw_population  = [self.cfs.fitness(i[1:])[0] for i in self.total_orgs if i[0]==-2]  # Pool all Hw individuals
        comp_population = [self.cfs.fitness(i[1:])[0] for i in self.total_orgs if i[0]==-1] # Pool all competent individuals

        if len(hw_population):
            m_hw = max(hw_population) # get the max fitness of all hw individuals
        else:
            m_hw =0

        if len(comp_population):
            m_cp = max(comp_population) # get max fitness of all competent individuals
        else:
            m_cp = 0

        return (m_hw, m_cp) # return a tuple of (max hw, max competent) fitness


    def run_ga(self):

        # main function running the genetic algorithm
        # returns: two lists: one tracking the fitnessess of both individuals over all generations and the other tracking their counts over all generations

        self.initialize() # intialize the mixture population

        self.hw_comp_fitness = []  # array to keep track of (best hw, best comp) fitness tuples in each generation
        self.counts = [] # array to keep track of the (number of hw, number of competent) counts in each generation

        generation = 1 # counter to keep track of generations elapsed

        while generation <= self.config['RUNS']: # Run until we hit the max number of generations

            self.hw_comp_fitness.append(self.calc_fitness_separately()) # Get the best fitnesses of hw and competent individuals and store them in the tracker list

            self.count_orgs() # count
            self.counts.append((self.hw_count, self.comp_count)) # store count in the tracker

            fittest_organisms, _ = self.cfs.combined_selection(self.total_orgs)  # Selection: hw based on hw fitness, competent based on competent fitness
            new_population = self.cfs.crossover_withoutInterbreeding(fittest_organisms)      # Cross-mutate
            mutated_population = self.cfs.mutation_flip(new_population)          # Point mutate

            self.total_orgs = mutated_population.copy()  # set the muated population as the new population

            generation +=1  # increment generations elapsed

        return self.hw_comp_fitness, self.counts



def plot(config, minorityFlag):

    # A global function to plot our results
    # a matrix of plots is constructed of size [number_of_cometencies x  mixture_ratios]
    # Plotting is possible only if simulated data is saved in the --savedir folder

    if minorityFlag:

        # Plot for Minority Experiment

        try:
            preva_hw = np.load(os.path.join(SAVE_DIR, 'Exp2a-hw_prevalence.npy'))
            preva_comp = np.load(os.path.join(SAVE_DIR, 'Exp2a-comp_prevalence.npy'))

        except FileNotFoundError:
            raise Exception('Save file for minority-Experiment not found')

        colors = ['crimson', 'olivedrab', 'teal', 'slateblue', 'orange', 'dimgrey']

        for bidx, b in enumerate(config['BubbleLimits']):
            mr_hw = np.mean(preva_hw[bidx, :, :], axis=0)
            vr_hw = np.std(preva_hw[bidx, : , :], axis =0)/np.sqrt(config['Loops'])

            plt.plot(np.arange(1, config['RUNS']+1), mr_hw, linestyle='--', linewidth=2.0, color=colors[bidx], label = 'Hardwired Population')
            plt.fill_between(np.arange(1,config['RUNS']+1), mr_hw-1*vr_hw, mr_hw+1*vr_hw, alpha =0.2, color = colors[bidx])

            mr_cmp = np.mean(preva_comp[bidx, : , :], axis = 0)
            vr_cmp = np.std(preva_comp[bidx, :, :], axis = 0)/np.sqrt(config['Loops'])

            plt.plot(np.arange(1, config['RUNS']+1), mr_cmp, linewidth=2.5, color=colors[bidx], label = 'Competent Population [Level {}]'.format(b))
            plt.fill_between(np.arange(1, config['RUNS']+1), mr_cmp-1*vr_cmp, mr_cmp+1*vr_cmp, alpha =0.2, color = colors[bidx])


        plt.xlabel('Generation')
        plt.ylabel("Percentage of Individuals (%)")
        plt.tight_layout()
        plt.legend(loc = 'lower right')
        plt.savefig(os.path.join(SAVE_DIR, 'Exp2a'), dpi=300)


    else:

        # Plot for Normal Mixed population experiement

        fig, axs = plt.subplots(len(config['BubbleLimits']), len(config['N_hw']), sharex=True, sharey=True, figsize=(10,13))
        row, col = 0,0

        try:
            preva_hw = np.load(os.path.join(SAVE_DIR, './hw.npy'))
            preva_comp = np.load(os.path.join(SAVE_DIR, './comp.npy'))

        except FileNotFoundError:
            raise Exception('Save file not found for mixed population experiment')


        for nb, bubble in enumerate(config['BubbleLimits']):

            for idx, (i, j) in enumerate(zip(config['N_hw'], config['N_comp'])):

                mr_hw = np.mean(preva_hw[nb, idx, :, :], axis=1)
                vr_hw = np.std(preva_hw[nb, idx, :, :], axis =1)/np.sqrt(config['Loops'])

                mr_cmp = np.mean(preva_comp[nb, idx, :, :], axis = 1)
                vr_cmp = np.std(preva_comp[nb, idx, :, :], axis = 1)/np.sqrt(config['Loops'])

                htrace, = axs[row,col].plot(range(1, config['RUNS']+1), mr_hw)
                axs[row, col].fill_between(range(1,config['RUNS']+1), mr_hw-1*vr_hw, mr_hw+1*vr_hw, alpha =0.2)

                ctrace, = axs[row,col].plot(range(1, config['RUNS']+1), mr_cmp)
                axs[row, col].fill_between(range(1, config['RUNS']+1), mr_cmp-1*vr_cmp, mr_cmp+1*vr_cmp, alpha =0.2)

                axs[row, col].legend()

                if col==0:
                    axs[row, col].set_ylabel('{} Swaps'.format(bubble))

                if row==0:
                    axs[row, col].set_title('{} HW - {} Competent'.format(i, j))

                col +=1

            col = 0
            row += 1

        fig.subplots_adjust(top=0.9)
        axs[0,1].legend([ctrace, htrace], ['Competent', 'Hardwired'], loc = 'upper center', bbox_to_anchor = (0.50, 1.35), fancybox =True, shadow = True)

        fig.supxlabel('Generation')
        fig.supylabel('Percentage of Individuals (%)')
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, 'Exp2'), dpi=300)

if __name__ == "__main__":

    # Argument parsing

    parser = argparse.ArgumentParser(description='Mixed population Experiment')
    parser.add_argument('--simulate', type=bool, default=False, help='Set True to run the experiment from scratch. If False, a plot is produced from saved data')
    parser.add_argument('--minorityExp', type =bool, default = False, help='Run the experiment with competent individuals being a minority (20% of the population) with different competency levels (check hyperparameters file for competency levels)')
    parser.add_argument('--hfile', type = str, default='./hyperparameters.json')
    parser.add_argument('--savedir', type = str, default='./MixedPopulationResults/', help = 'path to folder containing saved data (if plot), else destination to save data')

    args = parser.parse_args()

    # Global settings

    HYP_FILE_PATH = args.hfile
    SAVE_DIR = args.savedir

    if args.minorityExp:
        EXP_TYPE = 'mixed_population_minority'

    else:
        EXP_TYPE = 'mixed_population'

    with open(HYP_FILE_PATH, 'r') as f:
        config_data = json.load(f)

    config = config_data[EXP_TYPE]

    # We can run two types of experiments: 1. The original mixed population experiment, where individuals are mixed in different ratios and different competency levels.
    # 2. A single mixture proportion (competent 20%, hardwired 80%) is used. Different competency levels are compared to check at which level competent individuals dominate


    if args.simulate:

        if args.minorityExp: # Minority experiment 2

            preva_hw = np.zeros((len(config['BubbleLimits']),config['Loops'], config['RUNS']))
            preva_comp = np.zeros((len(config['BubbleLimits']), config['Loops'], config['RUNS']))

            for bidx, b in enumerate(config['BubbleLimits']):

                print('Running for Competency Level {}\n'.format(config['BubbleLimits'][bidx]))

                config['BubbleLimit'] = b
                cfs = HelperFuncs(config)
                run1 = MixedFit(config, cfs, n_hw = config['N_hw'], n_comp = config['N_comp'])

                for r in range(config['Loops']):

                    _, prevalence_counts  = run1.run_ga()

                    preva_hw[bidx, r] = [hw for hw, _ in prevalence_counts]
                    preva_comp[bidx, r] = [comp for _, comp in prevalence_counts]

            print('Saving....\n')

            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            np.save(os.path.join(SAVE_DIR, 'Exp2a-hw_prevalence'), preva_hw)
            np.save(os.path.join(SAVE_DIR, 'Exp2a-comp_prevalence'), preva_comp)

            print('Plotting...')
            plot(config, args.minorityExp)

        else:
            # Minority Experiment 1
            # Sanity check to ensure that the sum of N_hw + N_competent = Number of total organisms, in each of the mixture cases (see hyperparameter.json)

            for i, j in zip(config['N_hw'], config['N_comp']):
                if i+j != config['N_organisms']:
                    raise Exception("Number of Hardwired and Competent don't match the population size. Check your hyperparameters")


            preva_hw = np.zeros((len(config['BubbleLimits']), len(config['N_hw']), config['RUNS'], config['Loops']))        # create array to store simulations for every competency level, every mixture ratio, and every repeat
            preva_comp = np.zeros((len(config['BubbleLimits']), len(config['N_comp']), config['RUNS'], config['Loops']))    # create array to store simulations for every competency level, every mixture ratio, and every repeat

            print('Hyperparameters look good \n Starting Mixed Experiment...')

            # The order in which we store results is: competency_level > all mixture ratios > all repeats

            for bn, bubble in enumerate(config['BubbleLimits']):

                for idx, (i, j) in enumerate(zip(config['N_hw'], config['N_comp'])):

                    print('-'*10)
                    print('Running: {} HW and {} Competent individuals (Competency: {}) mixed together'.format(i, j, bubble))
                    print('-'*10)

                    for r in range(config['Loops']):

                        config['BubbleLimit'] = bubble

                        cfs = HelperFuncs(config)

                        # initalize an instance of the class with correct settings
                        run1 = MixedFit(config, cfs, i, j)

                        # run genetic algorithm
                        te_buff, all_cnts = run1.run_ga()

                        # store results

                        preva_hw[bn, idx, :, r] = [hw for hw, _ in all_cnts]
                        preva_comp[bn, idx, :, r] = [comp for _, comp in all_cnts]

            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)

            # save results in .npy files
            np.save(os.path.join(SAVE_DIR, './hw'), preva_hw)
            np.save(os.path.join(SAVE_DIR, './comp'), preva_comp)

            #plot
            print('Plotting...')
            plot(config, args.minorityExp)

    else:

        #plot
        print('Plotting...')
        plot(config, args.minorityExp)
