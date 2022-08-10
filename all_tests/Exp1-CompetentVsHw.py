import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from core_functions import HelperFuncs


# Global plot settings

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
sns.set_theme(style = "darkgrid")

plt.rcParams["figure.figsize"] = (7.1, 9)
sns.set_palette(sns.color_palette())

color_sets = ['orange', 'mediumorchid', 'teal', 'purple']
cats = [r"$\underset{0-Swaps}{Hardwired}$", r"$\underset{100-Swaps}{Competent}$"]

def run_ga (config, HW=True):
    # Main function running the genetic algorithm

    # Inputs: Configuration data and a boolean flag indicating if the simulation is for a HW population or not, Default = True
    # Returns: Array of best genotypic fitnessess in each generation, array of best phenotypic fitnessess in each generation, (Genomes of populations in all generations (dict), Post-Swapped Genomes (Competent) of populations in all generations (dict))

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE'])) # Initialize a population with settings from the configuration data
    fcs = HelperFuncs(config) # Initialize core functions required for the Genetic Algorithm


    # Initialize arrays to track fitnessess and store populations

    GTracker = [] # Track the best genotypic fitness in each generation
    CTracker = [] # Track the best phenotypic Fitness in each generation 

    genomic_collection = {} # Map each generation to the genome of a population
    phenotypic_collection = {} # Map each generation to the post-swapped genome(Phenotype) of a population


    for g in range(1, config['RUNS']+1):
        # Iterations of the genetic algorithm

        HWFitness = [fcs.fitness(org)[0] for org in init_population] # Get all Fitnesses
        GTracker.append(np.max(HWFitness)) # Store the best genotypic fitness

        genomic_collection[g] = init_population.copy() # Store genomes of a population (regardless of HW or Competent) for a particular generation
 
        if HW: # Do only if HW populaiton is set

            SPopulation, _ = fcs.selection(init_population, HWFitness) # Carry out Selection based on HW Fitnessess 

        else: # Do only if Competent Population is set

            C_population = fcs.bubble_sort(init_population) # Allow Cells Swap their values
            
            phenotypic_collection[g] = C_population.copy() # Store the Post-swapped Genomes(Phenotype) of a Population for a particular generation

            CFitness = [fcs.fitness(org)[0] for org in C_population] # Get Fitness of all individuals
            CTracker.append(np.max(CFitness)) # Store the max phenotypic fitness

            SPopulation, _ = fcs.selection(init_population, CFitness) # Carry out selection based on post-swapped Fitnessess

        ROPopulation = fcs.crossover_mutation(SPopulation) # Regardless of how selection occurs, repopulate with cross-over

        RFPopulation = fcs.mutation_flip(ROPopulation) # Carry out point mutations

        init_population = RFPopulation.copy() # Replace current population with the evolved one

    if HW:
        return GTracker, genomic_collection

    else:

        return GTracker, CTracker, (genomic_collection, phenotypic_collection) # Once done, return all trackers, and the pre-swapped and post-swapped genomes


def plot_all(bubbles, N_repeats, N_generations = 1000):

    # Main plotting function
    # Inputs: Configuration dict, Number of repeats to plot, Number of generations to plot 
    # Plotting is possible only if Simulated data is present in --save_dir

    # Load saved data
    hw_runs = np.load(os.path.join(SAVE_DIR, 'hwBubbleSort.npy'))
    comp_genome_runs = np.load(os.path.join(SAVE_DIR, 'compGenomeBubbleSort.npy'))
    comp_phenotype_runs = np.load(os.path.join(SAVE_DIR, 'compPhenotypeBubbleSort.npy'))

    hw_runs = hw_runs[:N_repeats, :N_generations]
    comp_genome_runs = comp_genome_runs[:, :N_repeats, :N_generations] 
    comp_phenotype_runs = comp_phenotype_runs[:, :N_repeats, :N_generations]

    # Plot hardwired fitnesses

    m_hw = np.mean(hw_runs, axis = 0)
    var_hw = np.std(hw_runs, axis = 0)/np.sqrt(config['Loops'])
    p1,  = plt.plot(range(1, len(m_hw)+1), m_hw, label='Hardwired Genotypic Fitness [No Swaps]', linestyle='--', color='black', linewidth=1.5)
    plt.fill_between(range(1, len(m_hw)+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2, color = 'black')


    lvls = [0.65, 0.75,  0.80, 0.9, 0.97, 1.0] # Check when each fitness curve reaches a particular threshold

    for l in lvls: 
        hw_time = N_generations - len(m_hw[m_hw>=l])
        print("Hw reaches {} at {}".format(l, hw_time))

    # Plot Genotypic and Phenotypic fitnessess of Competent populations

    for b in range(len(bubbles)):
        m_comp = np.mean(comp_genome_runs[b, : , :], axis = 0)
        var_comp = np.std(comp_genome_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        p1, = plt.plot(range(1, len(m_comp)+1), m_comp, label = 'Competent Genotypic Fitness [{} Swaps]'.format(bubbles[b]), linestyle='--',  color =color_sets[b])
        plt.fill_between(range(1, len(m_comp)+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2, color = color_sets[b])

        m_comp = np.mean(comp_phenotype_runs[b, : , :], axis = 0)
        var_comp = np.std(comp_phenotype_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        p1, = plt.plot(range(1, len(m_comp)+1), m_comp, label = 'Competent Phenotypic Fitness [{} Swaps]'.format(bubbles[b]), color = color_sets[b])
        plt.fill_between(range(1, len(m_comp)+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2, color = color_sets[b])
        print('*'*10)
        print('\n')

        for l in lvls: 
            cmp_time = N_generations - len(m_comp[m_comp>=l])
            print("comp({} bubbles) reaches {} at {}".format(bubbles[b], l, cmp_time+1))


    plt.xlabel('Generation')
    plt.ylabel('Fitness of the Best Individual')

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'Exp1'), dpi=300)
    plt.show()



if __name__ == '__main__':

    # Argument parsing

    parser = argparse.ArgumentParser(description='Single Population Experiment')
    parser.add_argument('--simulate', type=bool, default=False, help='Set True to run the experiment from scratch. If False, a plot is produced from saved data')
    parser.add_argument('--hfile', type = str, default='./hyperparameters.json')
    parser.add_argument('--savedir', type = str, default='./SinglePopulation/', help = 'path to folder containing saved data (if plot), else destination to save data')

    args = parser.parse_args()

    # Global settings

    HYP_FILE_PATH = args.hfile
    SAVE_DIR = args.savedir
    EXP_TYPE = 'single_population'

    with open(HYP_FILE_PATH, 'r') as f:
        config_data = json.load(f)

    config = config_data[EXP_TYPE]

    print('Experiment 1: Running HW versus Competent for {} set of competent levels: {}\n'.format(len(config['BubbleLimits']), config['BubbleLimits']))

    if args.simulate:

        # Prepare to run the genetic algorithm

        hw_runs = np.zeros((config['Loops'], config['RUNS'])) # Initialize array to store all HW population runs
        comp_genome_runs = np.zeros((len(config['BubbleLimits']), config['Loops'], config['RUNS']))  # Initialize array to store all genomes of competent runs
        comp_phenotype_runs = np.zeros((len(config['BubbleLimits']), config['Loops'], config['RUNS'])) # Initialize array to store all phenotypies of competent runs

        hw_runs_collection = {} # dict to store HW genomes
        comp_runs_collection = {} # dict to store competent genomes and phenotypes

        # Basically, we run multiple repeats of the following process: 
        # 1. Evolve a HW population
        # 2. Evolve each competent population
        # 3. Save their results

        for yu in range(config['Loops']):

            print('**'*10)
            print('In Run : {}'.format(yu))
            print('**'*10)
            print('\n')
            print('Hardwired Run...')
            
            fits, hw_genome_populations = run_ga(config, HW=True) # Evolve the HW population only
            hw_runs[yu, :] = fits # store the best genomic fitnessess over all generations
            hw_runs_collection[yu] = hw_genome_populations # Store the genomes of the HW population for a particular run


            comp_bubbles_collection = {} # dict to store the genotypes and phenotypes of each competent population

            for k, r in enumerate(config['BubbleLimits']): 

                print('Competency run: {} bubbles'.format(r))

                config['BubbleLimit'] = r

                cp_gen_fits, cp_cmp_fits, populations_tuple = run_ga(config, HW=False) 
                comp_genome_runs[k, yu, :], comp_phenotype_runs[k, yu, :] = cp_gen_fits, cp_cmp_fits # Store the genomic fitness, phenotypic fitness of a specific competent population

                comp_bubbles_collection[r] = populations_tuple # Store the pre-swap and post swap genomes tuple of each competent population


            comp_runs_collection[yu] = comp_bubbles_collection # Store the genomes of all competent populations for a specific run

        print('Saving Run {}... \n'.format(yu))

        if not os.path.exists(SAVE_DIR):
            print('Save directory not found, creating one...')
            os.makedirs(SAVE_DIR)

        np.save(os.path.join(SAVE_DIR, 'hwBubbleSort'), hw_runs)
        np.save(os.path.join(SAVE_DIR, 'compGenomeBubbleSort'), comp_genome_runs)
        np.save(os.path.join(SAVE_DIR, 'compPhenotypeBubbleSort'), comp_phenotype_runs) 
        np.save(os.path.join(SAVE_DIR, 'hwPopulations'), hw_runs_collection)
        np.save(os.path.join(SAVE_DIR, 'compPopulationsGC'), comp_runs_collection)

        plot_all(config['BubbleLimits'], N_repeats=config['Loops'], N_generations=config['RUNS']) 

    else:

        plot_all(config['BubbleLimits'], N_repeats=config['Loops'], N_generations=config['RUNS']) 
