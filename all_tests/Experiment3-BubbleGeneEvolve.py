from cProfile import label
import numpy as np
from core_functions import HelperFuncs
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


sns.set_theme(style = "darkgrid")
plt.rcParams["figure.figsize"] = (7, 7)
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

        RFPopulation = fcs.mutation_flip_incl_gene(ROPopulation, val = 400)

        #stPopulation_fov = fcs.mutation_flip_stressgene(RFPopulation, -2)
        # stPopulation_bubble = fcs.mutation_flip_stressgene(RFPopulation, -1)

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
    hw_runs = np.load('./weights/Hw_2genes.npy')
    comp_runs = np.load('./weights/Comp_2genes.npy')
    # fov_gene_tracker = np.load('./weights/fov_genetracker_2genes.npy')
    # stress_gene_tracker = np.load('./weights/stress_genetracker_2genes.npy')
    gene_tracker = np.load('./weights/genetracker_bubble.npy')
    low_genetracker_bubble = np.load('./weights/low_genetracker_bubble.npy')
    high_genetracker_bubble = np.load('./weights/high_genetracker_bubble.npy')
    # corrs = np.load('./weights/1segCorr.npy')

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
        ax2.plot(ms, label = 'Gene Value of the Best Individual', color = 'green')#, marker = 'x') 
        ax2.fill_between(range(100), ms-l_ms, h_ms, alpha = 0.2, label = 'Range of Gene Values', color='green')
        ax2.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])
        # for n, bin in enumerate(gene_splits):

        #     m = np.mean(bin)
        #     if n==0:
        #         # ax2.fill_between(bin-low_gene_splits[n], high_gene_splits[n], alpha = 0.2, label = 'Range of Highest and Lowest Gene values')
        #     else:
        #         ax2.plot(m)
        #         # ax2.fill_between(bin-low_gene_splits[n], high_gene_splits[n], alpha = 0.2)

    else:
        m_comp = np.mean(gene_tracker, axis = 0)
        ax2.plot(range(1, config['RUNS']+1), m_comp, label = 'Gene Value of the Best Individual', color = 'green')#, marker = 'x')
        ax2.fill_between(range(1, config['RUNS']+1), m_comp-low_gene_mean, high_gene_mean, alpha = 0.2, label = 'Range of gene values')

        ax2.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])


    corrs = np.nan_to_num(corrs)
    ax3.plot(corrs, label='Correlation of Genotypic and Phenotypic Fitness', color='purple', marker ='x')
    ax3.set_xticks(ticks = np.arange(0, 120, 20), labels = ['0', '200', '400', '600', '800', '1000'])

    # m_comp = np.mean(mean_genetracker_bubble, axis = 0)
    # var_comp = np.std(mean_genetracker_bubble, axis = 0)/np.sqrt(config['Loops'])
    # ax4.plot(range(1, config['RUNS']+1), m_comp, label = 'Average Competency (Evolving Swaps)', marker = 'x')



        # m_comp = np.mean(stress_gene_tracker[j, :, :], axis = 0)
        # var_comp = np.std(stress_gene_tracker[j, :, :], axis = 0)/np.sqrt(config['Loops'])
        # ax3.scatter(range(1, config['RUNS']+1), m_comp, label = 'Competent ({} swaps)'.format(bubbles[j]), marker='x')

    ax1.set(xlabel = 'Generation', ylabel='Fitness of best Individual (Evolving Swaps)')
    ax2.set(xlabel = 'Generation', ylabel='Number of Swaps')
    ax3.set(xlabel = 'Generation', ylabel='Correlation Value')

    ax1.text(-0.1, 1.15, 'B', transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    ax2.text(-0.1, 1.15, 'A', transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    ax3.text(-0.1, 1.15, 'C', transform=ax3.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
    


    # ax4.set(xlabel='Generation', ylabel='Average Competency within a Generation')
    #ax1.set_title('Fitness plot')
    #ax2.set_title('Preferred gene (Max fitness individual)')
    #ax2.legend(loc='upper right')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax4.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig('./final_figures/Exp3:A', dpi=300)
    # plt.show()

def get_max(orgs, config):
    fcs = HelperFuncs(config)
    fits = [fcs.fitness(i)[0] for i in orgs]
    mx = np.max(fits)
    indx = np.where(fits == mx)[0][0]
    best_indv = orgs[indx, :]
    return best_indv.reshape(1,-1)

def print_all(config):
    # hw_pop = np.load('./weights/genome_population.npy', allow_pickle=True)
    # all_changes = np.zeros((10, 51))
    # record_str = np.zeros((10, 1000))
    # record_func = np.zeros((10, 1000))
    # for i in range(10):
    #     change_array = np.zeros(51)
    #     pop = hw_pop.item().get(i)
    #     for j in range(1, 999):
    #         current_pop = np.array(pop[j]).reshape(100,51)
    #         next_pop = np.array(pop[j+1]).reshape(100, 51)


    #         best_current_pop = get_max(current_pop, config) 
    #         best_next_pop = get_max(next_pop, config) 
    #         res = np.subtract(best_current_pop, best_next_pop).reshape(-1)
    #         temp_change_array = [0 if q==0 else 1 for q in res]
    #         change_array += temp_change_array
    #         record_str[i, j] = np.mean(change_array[:-1]) 
    #         record_func[i, j] = change_array[-1]

    #     # all_changes[i] = change_array

    # stplot = np.mean(record_str, axis =0) 
    # fnplot = np.mean(record_func, axis = 0)
    # plt.plot(stplot, label ='Structural Genes ')
    # plt.plot(fnplot, label = 'Functional Gene')


    # np.save('strcrec', record_str)
    # np.save('funcrec', record_func)
    # np.save('change_array', all_changes)
    fig, (ax2, ax1) = plt.subplots(2,1, figsize=(9,14))
    plt.rcParams.update({'lines.markeredgewidth': 1})

    

    record_str = np.load('strcrec.npy')
    record_func = np.load('funcrec.npy')

    stplot = np.mean(record_str[:, :-1], axis =0) 
    st_stdv = np.std(record_str[:, :-1], axis = 0)/np.sqrt(10)
    ax1.plot(stplot, label ='Average of 50 Structural Genes', color = 'maroon')
    ax1.fill_between(range(1, 1000), stplot-st_stdv, stplot+st_stdv, alpha = 0.2, color = 'maroon')

    fnplot = np.mean(record_func[:, :-1], axis = 0)
    fn_stdv = np.std(record_func[:, :-1], axis = 0)/np.sqrt(10) 
    ax1.plot(fnplot, label = 'Competency Gene', color = 'teal')
    ax1.fill_between(range(1, 1000), fnplot-fn_stdv, fnplot+fn_stdv, alpha = 0.2, color = 'teal')
 
    # all_changes = np.load('./change_array.npy')
    # print(all_changes)

    # for i in range(51):
        # all_changes[i] = all_changes[i]/max(all_changes[i])
        # all_changes[i] = (all_changes[i]-np.mean(all_changes[i]))/np.std(all_changes[i])

    # all_changes = np.mean(all_changes, axis = 0).reshape(1,-1)
    bar_st = np.mean(record_str[:, :-1], axis =1)
    bar_fun = np.mean(record_func[:, :-1], axis =1)


    m_bar_st = np.mean(bar_st)
    m_bar_fun = np.mean(bar_fun)

    std_st = np.std(bar_st)
    std_fun = np.std(bar_fun)

    # structural = np.mean(all_changes[0][:-1])
    # functional = all_changes[0][-1]

    ax2.bar(['Structural Genes', 'Competency Gene'],[m_bar_st, m_bar_fun], yerr = [std_st, std_fun], color = ['maroon', 'teal'], capsize = 13)#,palette = ['maroon', 'teal'])
    # plt.ylabel('Number of times Changed')

    # ax = sns.heatmap(all_changes, linewidth =0.9, yticklabels=False) 
    # ax.set(xlabel = 'i', ylabel =' i')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Change Frequency')

    ax2.set_ylabel('Average Change Frequency')
    ax1.text(-0.1, 1.15, 'B', transform=ax1.transAxes,fontsize=17, fontweight='bold', va='top', ha='right')
    ax2.text(-0.1, 1.15, 'A', transform=ax2.transAxes,fontsize=17, fontweight='bold', va='top', ha='right')


    fig.tight_layout()
    fig.legend()
    plt.savefig('./genefrequency', dpi = 300)
    plt.show()
            

            
            
if __name__ == '__main__':
    config = {'LOW': 1,
                    'HIGH' : 51, 
                    'SIZE' : 50,
                    'N_organisms' : 100,
                    'RUNS' : 1000,
                    'Mutation_Probability' : 0.6,
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

    # hw_runs = np.zeros((config['Loops'], config['RUNS']))
    # comp_runs = np.zeros((config['Loops'], config['RUNS']))

    # #comp_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    # # fov_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    # # stress_gene_tracker = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    # bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))
    # low_bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))
    # high_bubble_genetracker = np.zeros((config['Loops'], config['RUNS']))

    # hw_ind_pop = {}
    # comp_ind_pop = {}
    
    # for yu in tqdm(range(config['Loops'])):
    #     print('In Run : {}'.format(yu))
    #     print('**'*10)
    #     print('\n')
    #     # fits = run_ga(config, HW=True)
    #     fits, gtrck, compfits, low_gtrck, high_gtrck, population_collections = run_ga(config, HW=False)
    #     hw_runs[yu, :] = fits

    #     # cp_fits, gtrck = run_ga(config, HW=False)
    #     comp_runs[yu, :] = compfits
    #     # fov_gene_tracker[k, yu, :] = gtrck[0]
    #     # stress_gene_tracker[k, yu, :] = gtrck[1]
    #     bubble_genetracker[yu, :] = gtrck

    #     low_bubble_genetracker[yu, :] = low_gtrck
    #     high_bubble_genetracker[yu, :] = high_gtrck

    #     hw_mean = np.mean(hw_runs, axis =0)
    #     comp_mean = np.mean(comp_runs, axis = 0)

    #     hw_ind_pop[yu] = population_collections[0]
    #     comp_ind_pop[yu] = population_collections[1]

    
    # np.save('./weights/Hw_2genes', hw_runs)
    # np.save('./weights/Comp_2genes', comp_runs)
    # # np.save('./weights/fov_genetracker_2genes', fov_gene_tracker)
    # # np.save('./weights/stress_genetracker_2genes', stress_gene_tracker)
    # np.save('./weights/genetracker_bubble', bubble_genetracker)
    # np.save('./weights/low_genetracker_bubble', low_bubble_genetracker)
    # np.save('./weights/high_genetracker_bubble', high_bubble_genetracker)
    #     # np.save('./weights/genome_population', population_collections[0])
    #     # np.save('./weights/Competent_population', population_collections[1])

    # np.save('./weights/genome_population', hw_ind_pop) 
    # np.save('./weights/Competent_population', comp_ind_pop)


    # plot_all(rns = config['Loops'], binned=True)
 
    print_all(config)
