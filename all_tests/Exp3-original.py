#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from core_functions import HelperFuncs


def run_ga (config, HW=False):

    # Main function running the genetic algorithm
    # Inputs: A configuration file, and indication if a competent population is used or not
    # Returns: The genomic fitnessess, competency-gene value of the best individual, the phenotypic fitnesses, the lowest competency gene value (for each generations), the highest competency gene values (for each generation), tuple of genomes and post-swapped genomes

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE'])) # Initialize a random population
    nswapGenes = np.random.randint(low = config['LOW'], high = 15, size= (config['N_organisms'], 1)) # Get random gene values (notice that the max value = 15)

    init_population = np.append(init_population, nswapGenes, axis = 1) # Combine the genes to create the evolvable population. The competency gene is at index 50 (the end of the array)

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []
    genetracker = []
    low_genetracker = []
    high_genetracker = []

    hw_collection = {}
    comp_collection = {}

    # Generation 0 is when the population is initialized. Generation 1 is when the population undergoes first selection

    for g in range(config['RUNS']):

        hw_collection[g] = init_population.copy()

        HWFitness = [fcs.fitness(org[:-1])[0] for org in init_population]
        HTracker.append(np.max(HWFitness))

        genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        low_genetracker.append(np.min([i[-1] for i in init_population]))
        high_genetracker.append(np.max([i[-1] for i in init_population]))

        if HW:

            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:

            C_population = fcs.bubble_sortevolve(init_population)
            comp_collection[g] = C_population.copy()

            CFitness = [fcs.fitness(org[:-1])[0] for org in C_population]
            CTracker.append(np.max(CFitness))

            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip_incl_gene(ROPopulation, max_val_to_mutate = 500) # Notice the very high mutation value

        init_population = RFPopulation.copy()

    return HTracker, genetracker, CTracker, low_genetracker, high_genetracker, (hw_collection, comp_collection)


def plot_all(rns, binned=False):

    # Main plot function; the following graphs are plot:

    # 1. Competency-Gene value of the Best Individual w/ shaded bounds representing highest and lowest competency-gene values
    # 2. Genotypic vs phenotypic Fitnessess of the best individual
    # 3. Correlation between genotypic and phenotypic fitness

    sns.set_theme(style = "darkgrid")
    plt.rcParams["figure.figsize"] = (7, 7)
    sns.set_palette(sns.color_palette())

    fig, (ax2, ax1, ax3) = plt.subplots(3, 1, figsize =(8, 12))

    try:
        hw_runs = np.load(os.path.join(SAVE_DIR, 'GenotypicFitness.npy'))
        comp_runs = np.load(os.path.join(SAVE_DIR, 'PhenotypicFitness.npy'))
        gene_tracker = np.load(os.path.join(SAVE_DIR, 'CompetencyGeneValue.npy'))
        low_genetracker_bubble = np.load(os.path.join(SAVE_DIR, 'MinCompGeneValue.npy'))
        high_genetracker_bubble = np.load(os.path.join(SAVE_DIR, 'MaxCompGeneValue.npy'))

    except FileNotFoundError:
        raise Exception('Save files not found')

    hw_runs = hw_runs[:rns, :]
    comp_runs = comp_runs[:rns, :]
    gene_tracker = gene_tracker[:rns, :]
    low_genetracker_bubble = low_genetracker_bubble[:rns, :]
    high_genetracker_bubble = high_genetracker_bubble[:rns, :]

    hw_mean = np.mean(hw_runs, axis =0)
    comp_mean = np.mean(comp_runs, axis =0)
    gene_mean = np.mean(gene_tracker, axis = 0)
    low_gene = np.mean(low_genetracker_bubble, axis = 0)
    high_gene = np.mean(high_genetracker_bubble, axis =0)

    hw_mean_splits = np.split(hw_mean, config['RUNS']//10)
    comp_mean_splits = np.split(comp_mean, config['RUNS']//10)
    gene_splits = np.split(gene_mean, config['RUNS']//10)
    low_gene_splits = np.split(low_gene, config['RUNS']//10)
    high_gene_splits = np.split(high_gene, config['RUNS']//10)

    corrs = np.array([np.corrcoef(i, j)[0,1] for i, j in zip(hw_mean_splits, comp_mean_splits)])

    m_hw = np.mean(hw_runs, axis = 0)
    var_hw = np.std(hw_runs, axis = 0)/np.sqrt(config['Loops'])
    ax1.plot(range(1, config['RUNS']+1), m_hw, label='Genotypic Fitness')
    ax1.fill_between(range(1, config['RUNS']+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2)

    m_comp = np.mean(comp_runs, axis = 0)
    var_comp = np.std(comp_runs, axis = 0)/np.sqrt(config['Loops'])
    ax1.plot(range(1, config['RUNS']+1), m_comp, label = 'Phenotypic Fitness')
    ax1.fill_between(range(1, config['RUNS']+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2)

    low_gene_mean = np.mean(low_genetracker_bubble, axis=0)
    high_gene_mean = np.mean(high_genetracker_bubble, axis =0)

    if binned == True:

        ms = np.array([np.mean(bin) for bin in gene_splits])
        l_ms = np.array([np.mean(bin) for bin in low_gene_splits])
        h_ms = np.array([np.mean(bin) for bin in high_gene_splits])
        ax2.plot(ms, label = 'Gene Value of the Best Individual', color = 'green')
        ax2.fill_between(range(len(ms)), ms-l_ms, h_ms, alpha = 0.2, label = 'Range of Gene Values', color='green')
        ax2.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])

    else:
        m_comp = np.mean(gene_tracker, axis = 0)
        ax2.plot(range(1, config['RUNS']+1), m_comp, label = 'Gene Value of the Best Individual', color = 'green')
        ax2.fill_between(range(1, config['RUNS']+1), m_comp-low_gene_mean, high_gene_mean, alpha = 0.2, label = 'Range of gene values')
        ax2.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])


    corrs = np.nan_to_num(corrs)
    ax3.plot(corrs, label='Correlation of Genotypic and Phenotypic Fitness', color='purple', marker ='x')
    ax3.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])


    ax1.set(xlabel = 'Generation', ylabel='Fitness of best Individual (Evolving Swaps)')
    ax2.set(xlabel = 'Generation', ylabel='Number of Swaps')
    ax3.set(xlabel = 'Generation', ylabel='Correlation Value')

    ax1.text(-0.1, 1.15, 'B', transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    ax2.text(-0.1, 1.15, 'A', transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    ax3.text(-0.1, 1.15, 'C', transform=ax3.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')


    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'Exp3-Fitness'), dpi=300)

def get_max(orgs, config):
    fcs = HelperFuncs(config)
    fits = [fcs.fitness(i)[0] for i in orgs]
    mx = np.max(fits)
    indx = np.where(fits == mx)[0][0]
    best_indv = orgs[indx, :]
    return best_indv.reshape(1,-1)

def plotGeneChangeFrequency(config):

    hw_pop = np.load(os.path.join(SAVE_DIR, 'Genomes.npy'), allow_pickle=True)
    record_str = np.zeros((config['Loops'], config['RUNS']))
    record_func = np.zeros((config['Loops'], config['RUNS']))
    for i in range(config['Loops']):
        change_array = np.zeros(config['SIZE'] + 1)
        pop = hw_pop.item().get(i)
        for j in range(1, config['RUNS']-1):
            current_pop = np.array(pop[j]).reshape(config['N_organisms'], config['SIZE']+1)
            next_pop = np.array(pop[j+1]).reshape(config['N_organisms'], config['SIZE'] +1)


            best_current_pop = get_max(current_pop, config)
            best_next_pop = get_max(next_pop, config)
            res = np.subtract(best_current_pop, best_next_pop).reshape(-1)
            temp_change_array = [0 if q==0 else 1 for q in res]
            change_array += temp_change_array
            record_str[i, j] = np.mean(change_array[:-1])
            record_func[i, j] = change_array[-1]

    fig, (ax2, ax1) = plt.subplots(2,1, figsize=(9,14))
    plt.rcParams.update({'lines.markeredgewidth': 1})


    stplot = np.mean(record_str[:, :-1], axis =0)
    st_stdv = np.std(record_str[:, :-1], axis = 0)
    ax1.plot(stplot, label ='Average of 50 Structural Genes', color = 'maroon')
    ax1.fill_between(range(1, config['RUNS']), stplot-st_stdv, stplot+st_stdv, alpha = 0.2, color = 'maroon')

    fnplot = np.mean(record_func[:, :-1], axis = 0)
    fn_stdv = np.std(record_func[:, :-1], axis = 0)
    ax1.plot(fnplot, label = 'Competency Gene', color = 'teal')
    ax1.fill_between(range(1, config['RUNS']), fnplot-fn_stdv, fnplot+fn_stdv, alpha = 0.2, color = 'teal')

    bar_st = np.mean(record_str[:, :-1], axis =1)
    bar_fun = np.mean(record_func[:, :-1], axis =1)

    m_bar_st = np.mean(bar_st)
    m_bar_fun = np.mean(bar_fun)

    std_st = np.std(bar_st)
    std_fun = np.std(bar_fun)

    ax2.bar(['Structural Genes', 'Competency Gene'],[m_bar_st, m_bar_fun], yerr = [std_st, std_fun], color = ['maroon', 'teal'], capsize = 13)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Change Frequency')
    ax1.text(-0.1, 1.15, 'B', transform=ax1.transAxes,fontsize=17, fontweight='bold', va='top', ha='right')

    ax2.set_ylabel('Average Change Frequency')
    ax2.text(-0.1, 1.15, 'A', transform=ax2.transAxes,fontsize=17, fontweight='bold', va='top', ha='right')


    fig.tight_layout()
    fig.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'Exp3-changeFrequency'), dpi = 300)


def plot_topo():

    sns.set_theme(style = "darkgrid")
    sns.set_palette(sns.color_palette())

    hw_runs = np.load(os.path.join(SAVE_DIR, 'GenotypicFitness.npy'))
    comp_runs = np.load(os.path.join(SAVE_DIR, 'PhenotypicFitness.npy'))
    gene_tracker = np.load(os.path.join(SAVE_DIR, 'CompetencyGeneValue.npy'))
    low_genetracker_bubble = np.load(os.path.join(SAVE_DIR, 'MinCompGeneValue.npy'))
    high_genetracker_bubble = np.load(os.path.join(SAVE_DIR, 'MaxCompGeneValue.npy'))

    hw_mean = np.mean(hw_runs, axis =1)
    comp_mean = np.mean(comp_runs, axis =1)
    gene_mean = np.mean(gene_tracker, axis = 1)
    # low_gene = np.mean(low_genetracker_bubble, axis = 1)
    # high_gene = np.mean(high_genetracker_bubble, axis =1)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    reach_list = []
    st_gene_list = []
    xvals =[]
    yvals =[]

    for n, pair in enumerate(total_list):
        top_genes = gene_mean[n]
        ax.plot(top_genes, label = f'Strgncy: {pair[0]}, Mut_prob: {pair[1]}')
        stable_gene_value = top_genes[-10:].mean()
        st_gene_list.append(stable_gene_value)
        xvals.append(pair[0])
        yvals.append(pair[1])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Gene value of the Best individual')
    plt.draw()



    # for n, pair in enumerate(total_list):
    #     temp = comp_mean[n]
    #     stable_gene_value = top_genes[-10:].mean()
    #     try:
    #         reached_at = np.where(temp >=F_TO_REACH)[0][0]
    #     except IndexError:
    #         print('Fitness of {} was not reached'.format(F_TO_REACH))
    #         print('Skipping plot')
    #         sys.exit()
    #     reach_list.append(reached_at +1)
    #     xvals.append(pair[0])
    #     yvals.append(pair[1])

    # ax1.plot3D(xvals, yvals, reach_list)

    sns.set_palette(sns.color_palette())
    ax = fig.add_subplot(2,1,2, projection='3d')
    ax.scatter3D(xvals, yvals, st_gene_list, c=sns.color_palette(n_colors=9))
    ax.set_xlabel('Stringency')
    ax.set_ylabel('Mutation_probability')
    ax.set_zlabel('Stable Gene Value of the Best Individual')

    fig.tight_layout()
    leg = fig.legend()
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    bb.x1 +=1.5
    bb.y1 -= 1.4
    # bb.y0 -=1
    leg.set_bbox_to_anchor(bb, transform =ax.transAxes)


    plt.savefig(os.path.join(SAVE_DIR, 'Exp3:Topographic_Map'), dpi = 300)
    plt.show()



if __name__ == '__main__':

    # Argument parsing

    parser = argparse.ArgumentParser(description='Gene Evolution Experiment')
    parser.add_argument('--simulate', type=bool, default=False, help='Set True to run the experiment from scratch. If False, a plot is produced from saved data')
    parser.add_argument('--hfile', type = str, default='./hyperparameters.json')
    parser.add_argument('--savedir', type = str, default='./EvolvableCompetencyOriginalResults/', help = 'path to folder containing saved data (if plot), else, it indicates the destination to save simulated data')
    parser.add_argument('--plotType', type=str, default='fitness', choices = ['fitness', 'frequency'],  help='Set True to run the experiment from scratch. If False, a plot is produced from saved data')


    args = parser.parse_args()

    # Global settings

    HYP_FILE_PATH = args.hfile
    SAVE_DIR = args.savedir
    EXP_TYPE = 'single_population'
    N_VALUES = 1
    F_TO_REACH = 1.0

    with open(HYP_FILE_PATH, 'r') as f:
        config_data = json.load(f)

    config = config_data[EXP_TYPE]

    stringency_list = [0.1]
    mutprob_list = [0.6]

    total_list = [(i,j) for j in mutprob_list for i in stringency_list]

    if args.simulate:

        # Initialize and Run

        # Note: This experiment involved the competency population only. Instead of having a fixed competency, we define a competency gene and allow it to be changed by evolution

        hw_runs = np.zeros((len(total_list), config['Loops'], config['RUNS']))                       # Stores Genomic fitness of the best individual from a competent population
        comp_runs = np.zeros((len(total_list), config['Loops'], config['RUNS']))                       # Stores Genomic fitness of the best individual from a competent population

        bubble_genetracker = np.zeros((len(total_list), config['Loops'], config['RUNS']))                       # Stores Genomic fitness of the best individual from a competent population

        low_bubble_genetracker = np.zeros((len(total_list), config['Loops'], config['RUNS']))                       # Stores Genomic fitness of the best individual from a competent population

        high_bubble_genetracker = np.zeros((len(total_list), config['Loops'], config['RUNS']))                       # Stores Genomic fitness of the best individual from a competent population

        hw_ind_pop = {}     # dictionary to store genomes of a competent population in every generation
        comp_ind_pop = {}   # dictionary to store post_swapped genes of a competent population

        for pircnts, uval in enumerate(total_list):
            print('---'*10)
            print('RUNNING | Stringency Val: {} | Mutation_prob: {}'.format(uval[0], uval[1]))

            config['Stringency'] = uval[0]
            config['Mutation_Probability'] = uval[1]

            hw_pop = {}
            comp_pop = {}

            for yu in tqdm(range(config['Loops'])): # Re-run the genetic algorithm many times

                print('Running Loop : {}/{}'.format(yu+1, config['Loops']))
                print('\n')

                fits, gtrck, compfits, low_gtrck, high_gtrck, population_collections = run_ga(config, HW=False) # Run one instance of the genetic algorithm with our settings

                hw_runs[pircnts, yu, :] = fits            # Store the best genomic fitnessess for all generations
                comp_runs[pircnts, yu, :] = compfits      # Store the best post-swapped fitnessess for all generations

                bubble_genetracker[pircnts, yu, :] = gtrck           # Store the gene-values of the best individual in every generations
                low_bubble_genetracker[pircnts, yu, :] = low_gtrck   # Store the lowest gene-vale for every generation
                high_bubble_genetracker[pircnts, yu, :] = high_gtrck # Store the highest gene-value for every generation

                hw_pop[yu] = population_collections[0]     # Store the genomes of a population (for all generations)
                comp_pop[yu] = population_collections[1]   # Sore the post-swapped genomes of a population (for all generations)

            hw_ind_pop[pircnts] = hw_pop
            comp_ind_pop[pircnts] = comp_pop


        print('Saving...')

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        np.save(os.path.join(SAVE_DIR, 'GenotypicFitness'), hw_runs)
        np.save(os.path.join(SAVE_DIR, 'PhenotypicFitness'), comp_runs)

        np.save(os.path.join(SAVE_DIR, 'CompetencyGeneValue'), bubble_genetracker)
        np.save(os.path.join(SAVE_DIR, 'MinCompGeneValue'), low_bubble_genetracker)
        np.save(os.path.join(SAVE_DIR, 'MaxCompGeneValue'), high_bubble_genetracker)

        np.save(os.path.join(SAVE_DIR, 'Genomes'), hw_ind_pop)
        np.save(os.path.join(SAVE_DIR, 'PostSwappedGenomes'), comp_ind_pop)

        print('Plotting...')

        # if args.plotType == 'fitness':

        #     plot_all(rns = config['Loops'], binned=True)

        # else:

        #     plotGeneChangeFrequency(config)

        plot_topo()

    else:

        plot_topo()

        print('Plotting...')

#         if args.plotType == 'fitness':

#             plot_all(rns = config['Loops'], binned=True)

#         else:

#lotGeneChangeFrequency(config)
