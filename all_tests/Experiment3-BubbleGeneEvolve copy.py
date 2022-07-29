from cProfile import label
import numpy as np
from core_functions import HelperFuncs
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


sns.set_theme(style = "darkgrid")
plt.rcParams["figure.figsize"] = (9, 11)
sns.set_palette(sns.color_palette())


def run_ga (config, HW): 
    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    #fov_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))
    #stress_genes = np.random.randint(low = config['LOW'], high = config['HIGH']//2, size= (config['N_organisms'], 1))

    nswapGenes = np.random.randint(low = config['LOW'], high = 15, size= (config['N_organisms'], 1))

    # init_population = np.append(init_population, fov_genes, axis = 1)
    # init_population = np.append(init_population, stress_genes, axis = 1)

    init_population = np.append(init_population, nswapGenes, axis = 1)

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []
    # fov_genetracker = []
    # stress_genetracker =[]
    genetracker = []

    low_genetracker = []
    high_genetracker = []

    hw_collection = {}
    comp_collection = {}

    for g in range(1, config['RUNS']+1):
        hw_collection[g] = init_population.copy()
        # HWFitness = [fcs.fitness(org[:-2])[0] for org in init_population]

        HWFitness = [fcs.fitness(org[:-1])[0] for org in init_population]

        HTracker.append(np.max(HWFitness))

        # fov_genetracker.append(init_population[np.argmax(HWFitness), :][-2])
        # stress_genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        genetracker.append(init_population[np.argmax(HWFitness), :][-1])

        low_genetracker.append(np.min([i[-1] for i in init_population]))
        high_genetracker.append(np.max([i[-1] for i in init_population]))


        # C_population = fcs.stress_bubble_reorg_fieldgene(init_population)
 

        if HW:
            # print('HW Selection RUN, shape: {}'.format(init_population.shape))
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            # print('Competent Selection RUN')
            C_population = fcs.bubble_sortevolve(init_population)
            comp_collection[g] = C_population.copy()


            CFitness = [fcs.fitness(org[:-1])[0] for org in C_population]

            CTracker.append(np.max(CFitness))

            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        # if g >=100:
        #     RFPopulation = fcs.mutation_flip_incl_gene(ROPopulation, val = 5)
        # else:

        if g == 200:
            # print('Scrambling')
            print('locking down')

            init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

            orig_genes = np.array([i[-1] for i in ROPopulation]).reshape(-1, 1)

            # init_population = np.append(init_population, fov_genes, axis = 1)
            # init_population = np.append(init_population, stress_genes, axis = 1)

            init_population = np.append(init_population, orig_genes, axis = 1)

        else:

            RFPopulation = fcs.mutation_flip_incl_gene(ROPopulation, val = 400)
            init_population = RFPopulation.copy() #stPopulation_stress.copy()
        
        # print('Run shape: {}'.format(init_population.shape))

    if HW:
        print('Plotting HW')
        #ax1.plot(HTracker, label='Hardwired')
        return HTracker

    else:
        print('Plotting Comp\n')
        #ax2.scatter(range(1, len(genetracker)+1), genetracker, label='Gene; for {} bubble cycles'.format(config['BubbleLimit']))
        #ax1.plot(CTracker, label = 'Competent ({} swaps)'.format(config['BubbleLimit']))
        # return HTracker, (fov_genetracker, stress_genetracker)
        return HTracker, genetracker, CTracker, low_genetracker, high_genetracker, (hw_collection, comp_collection)


def plot_all(rns, binned=False):
    fig, (ax2, ax1, ax3) = plt.subplots(3, 1, figsize =(8, 12))
    hw_runs = np.load('./fake_weights/Hw_2genes.npy')
    comp_runs = np.load('./fake_weights/Comp_2genes.npy')
    # fov_gene_tracker = np.load('./fake_weights/fov_genetracker_2genes.npy')
    # stress_gene_tracker = np.load('./fake_weights/stress_genetracker_2genes.npy')
    gene_tracker = np.load('./fake_weights/genetracker_bubble.npy')
    low_genetracker_bubble = np.load('./fake_weights/low_genetracker_bubble.npy')
    high_genetracker_bubble = np.load('./fake_weights/high_genetracker_bubble.npy')
    # corrs = np.load('./fake_weights/1segCorr.npy')

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
        ax2.plot(ms, label = 'Gene value of Best individual (10 generation bins)', color = 'green')#, marker = 'x') 
        ax2.fill_between(range(len(ms)), ms-l_ms, h_ms, alpha = 0.2, label = 'Range of Gene values (10 generation bins)', color='green')
        ax2.set_xticks(ticks = np.arange(0, 225, 25), labels = ['0', '250', '500', '750', '1000', '1250', '1500', '1750', '2000'])
        # for n, bin in enumerate(gene_splits):

        #     m = np.mean(bin)
        #     if n==0:
        #         # ax2.fill_between(bin-low_gene_splits[n], high_gene_splits[n], alpha = 0.2, label = 'Range of Highest and Lowest Gene values')
        #     else:
        #         ax2.plot(m)
        #         # ax2.fill_between(bin-low_gene_splits[n], high_gene_splits[n], alpha = 0.2)

    else:
        m_comp = np.mean(gene_tracker, axis = 0)
        ax2.plot(range(1, config['RUNS']+1), m_comp, label = 'Gene value of Best individual', color = 'green')#, marker = 'x')
        ax2.fill_between(range(1, config['RUNS']+1), m_comp-low_gene_mean, high_gene_mean, alpha = 0.2, label = 'Range of gene values', color = 'green')
        ax2.set_xticks(ticks = np.arange(0, 225, 25), labels = ['0', '250', '500', '750', '1000', '1250', '1500', '1750', '2000'])


    corrs = np.nan_to_num(corrs)
    ax3.plot(corrs, label='Correlation of Genotypic and Phenotypic Fitness (10 generation bins)', color='purple', marker ='x')
    ax3.set_xticks(ticks = np.arange(0, 225, 25), labels = ['0', '250', '500', '750', '1000', '1250', '1500', '1750', '2000'])

    # m_comp = np.mean(mean_genetracker_bubble, axis = 0)
    # var_comp = np.std(mean_genetracker_bubble, axis = 0)/np.sqrt(config['Loops'])
    # ax4.plot(range(1, config['RUNS']+1), m_comp, label = 'Average Competency (Evolving Swaps)', marker = 'x')



        # m_comp = np.mean(stress_gene_tracker[j, :, :], axis = 0)
        # var_comp = np.std(stress_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        # ax3.scatter(range(1, config['RUNS']+1), m_comp, label = 'Competent ({} swaps)'.format(bubbles[j]), marker='x')

    ax1.set(xlabel = 'Generation', ylabel='Fitness of best Individual (Evolving Swaps)')
    ax2.set(xlabel = 'Generation', ylabel='Number of swaps')
    ax3.set(xlabel = 'Generation', ylabel='Correlation value')
    # ax4.set(xlabel='Generation', ylabel='Average Competency within a Generation')
    #ax1.set_title('Fitness plot')
    #ax2.set_title('Preferred gene (Max fitness individual)')
    #ax2.legend(loc='upper right')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax4.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('exp3:reaching1_0_custom')#, dpi=300)
    # plt.show()




if __name__ == '__main__':
    config = {'LOW': 1,
                    'HIGH' : 51, 
                    'SIZE' : 50,
                    'N_organisms' : 100,
                    'RUNS' : 2000,
                    'Mutation_Probability' : 0.6,
                    #'Mutation_Probability_stress' : 0.95,
                    'LIMIT': 50,
                    'MAX_FIELD' : 50,
                    'MODE': 'normal',
                    'REORG_TYPE': 3,
                    'ExpFitness': True,
                    'BubbleLimit': 0,
                    #'viewfield': 1,
                    'Loops': 10,
        }

    #bubbles = np.array([1, 20, 100, 400])#np.arange(0, 250, 50)
    #plot_all(bubbles)

    hw_runs = np.zeros((config['Loops'], config['RUNS']))
    comp_runs = np.zeros((config['Loops'], config['RUNS']))

    #comp_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    # fov_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    # stress_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))
    low_bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))
    high_bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))

    hw_ind_pop = {}
    comp_ind_pop = {}
    
    for yu in tqdm(range(config['Loops'])):
        print('In Run : {}'.format(yu))
        print('**'*10)
        print('\n')
        # fits = run_ga(config, HW=True)
        fits, gtrck, compfits, low_gtrck, high_gtrck, population_collections = run_ga(config, HW=False)
        hw_runs[yu, :] = fits

        # cp_fits, gtrck = run_ga(config, HW=False)
        comp_runs[yu, :] = compfits
        # fov_gene_tracker[k, yu, :] = gtrck[0]
        # stress_gene_tracker[k, yu, :] = gtrck[1]
        bubble_genetracker[yu, :] = gtrck

        low_bubble_genetracker[yu, :] = low_gtrck
        high_bubble_genetracker[yu, :] = high_gtrck

        hw_mean = np.mean(hw_runs, axis =0)
        comp_mean = np.mean(comp_runs, axis = 0)

        hw_ind_pop[yu] = population_collections[0]
        comp_ind_pop[yu] = population_collections[1]

    
    np.save('./fake_weights/Hw_2genes', hw_runs)
    np.save('./fake_weights/Comp_2genes', comp_runs)
    # np.save('./fake_weights/fov_genetracker_2genes', fov_gene_tracker)
    # np.save('./fake_weights/stress_genetracker_2genes', stress_gene_tracker)
    np.save('./fake_weights/genetracker_bubble', bubble_genetracker)
    np.save('./fake_weights/low_genetracker_bubble', low_bubble_genetracker)
    np.save('./fake_weights/high_genetracker_bubble', high_bubble_genetracker)
        # np.save('./fake_weights/genome_population', population_collections[0])
        # np.save('./fake_weights/Competent_population', population_collections[1])

    np.save('./fake_weights/genome_population', hw_ind_pop) 
    np.save('./fake_weights/Competent_population', comp_ind_pop)


    plot_all(rns = config['Loops'], binned=True)
