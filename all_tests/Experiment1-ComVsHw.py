from core_functions import HelperFuncs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.lines as mlines

import matplotlib.patches as mpatches

import seaborn as sns
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

sns.set_theme(style = "darkgrid")
plt.rcParams["figure.figsize"] = (7.1, 9)
sns.set_palette(sns.color_palette())
# SEEDS = [0, 4847, 2390234, 982, 10293, 23424, 5875, 2365437569, 86874, 3433, 123498543]
color_sets = ['orange', 'mediumorchid', 'teal', 'purple']
cats = [r"$\underset{0-Swaps}{Hardwired}$", r"$\underset{100-Swaps}{Competent}$"]

def run_ga (config, HW):

    init_population = np.random.randint(low = config['LOW'], high=config['HIGH'], size=(config['N_organisms'], config['SIZE']))

    fcs = HelperFuncs(config)

    HTracker = []
    CTracker = []
    hw_collection = {}
    comp_collection = {}

    for g in range(1, config['RUNS']+1):
        hw_collection[g] = init_population.copy()
        HWFitness = [fcs.fitness(org)[0] for org in init_population]

        HTracker.append(np.max(HWFitness))
 
        if HW:
            SPopulation, _ = fcs.selection(init_population, HWFitness)

        else:
            C_population = fcs.bubble_sort(init_population)

            comp_collection[g] = C_population.copy()

            CFitness = [fcs.fitness(org)[0] for org in C_population]

            CTracker.append(np.max(CFitness))

            SPopulation, _ = fcs.selection(init_population, CFitness)

        ROPopulation = fcs.crossover_mutation(SPopulation)

        RFPopulation = fcs.mutation_flip(ROPopulation)

        init_population = RFPopulation.copy()



    if HW:
        # print('Plotting HW')
        return HTracker, hw_collection

    else:
        # print('Plotting Comp')
        return HTracker, CTracker, (hw_collection, comp_collection)

def plot_all(bubbles, tar, max_n =1000):

    plotcollection = []
    hw_runs = np.load('./exp1/Hw_bubblesort.npy')
    comp_genome_runs = np.load('./exp1/Comp_genome_bubblesort.npy')
    comp_phenotype_runs = np.load('./exp1/Comp_phenotype_bubblesort.npy')

    hw_runs = hw_runs[:tar, :max_n]
    comp_genome_runs = comp_genome_runs[:, :tar, :max_n] 
    comp_phenotype_runs = comp_phenotype_runs[:, :tar, :max_n]

    m_hw = np.mean(hw_runs, axis = 0)
    var_hw = np.std(hw_runs, axis = 0)/np.sqrt(config['Loops'])
    p1,  = plt.plot(range(1, len(m_hw)+1), m_hw, label='Hardwired Genotypic Fitness [No Swaps]', linestyle='--', color='black', linewidth=1.5)
    plt.fill_between(range(1, len(m_hw)+1), m_hw-2*var_hw, m_hw+2*var_hw, alpha = 0.2, color = 'black')

    plotcollection.append(p1)
    lvls = [0.65, 0.75,  0.80, 0.9, 0.97, 1.0]

    pltgencoll = []
    pltphencoll = []

    for l in lvls: 
        hw_time = max_n - len(m_hw[m_hw>=l])
        print("Hw reaches {} at {}".format(l, hw_time))

    for b in range(len(bubbles)):
        # if bubbles[b] ==100:
        # m_comp = np.mean(comp_genome_runs[b, : , :], axis = 0)
        # var_comp = np.std(comp_genome_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        # p1, = plt.plot(range(1, len(m_comp)+1), m_comp, label = 'Competent Genotypic Fitness [{} Swaps]'.format(bubbles[b]), linestyle='--',  color =color_sets[b])
        # plt.fill_between(range(1, len(m_comp)+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2, color = color_sets[b])
        # pltgencoll.append(p1)

        m_comp = np.mean(comp_phenotype_runs[b, : , :], axis = 0)
        var_comp = np.std(comp_phenotype_runs[b, :, :], axis = 0)/np.sqrt(config['Loops'])
        p1, = plt.plot(range(1, len(m_comp)+1), m_comp, label = 'Competent Phenotypic Fitness [{} Swaps]'.format(bubbles[b]), color = color_sets[b])
        plt.fill_between(range(1, len(m_comp)+1), m_comp-2*var_comp, m_comp + 2*var_comp, alpha = 0.2, color = color_sets[b])
        pltphencoll.append(p1)
        print('*'*10)
        print('\n')

        for l in lvls: 
            cmp_time = max_n - len(m_comp[m_comp>=l])
            print("comp({} bubbles) reaches {} at {}".format(bubbles[b], l, cmp_time))

    # all_h_times ={}
    # for l in lvls:
    #     times = []
    #     for h_run in hw_runs:
    #         hw_time = max_n-len(h_run[h_run>=l])
    #         times.append(hw_time)
    #     all_h_times[l] = times

    # all_bubblec_times ={}
    # for b in range(len(bubbles)):
    #     all_c_times = {}
    #     for l in lvls:
    #         cTIMES = []
    #         for c_run in comp_phenotype_runs[b]:
    #             cmp_time = max_n-len(c_run[c_run>=l])
    #             cTIMES.append(cmp_time)

    #         all_c_times[l] = cTIMES
    #     all_bubblec_times[bubbles[b]] = all_c_times

    # ptfor = 0.65
    # for ptfor in lvls:
    #     print(ptfor)
    #     print(np.mean(all_h_times[ptfor]))
    #     print(np.var(all_h_times[ptfor]))
    #     print(np.mean(all_bubblec_times[100][ptfor]))
    #     print(np.var(all_bubblec_times[100][ptfor]))
    #     print('*'*10)

    # plt.hist(all_h_times[ptfor], bins = 30, label ='Hw')
    # plt.hist(all_bubblec_times[100][ptfor], bins = 30, label = 'Comp')
    ha = mlines.Line2D([], [], marker='None', linestyle='None')
    hb = mlines.Line2D([], [], marker='None', linestyle='None')
    hblank = mpatches.Patch(visible=False)

    pltcomb = list(zip(pltgencoll, pltphencoll))
    plt.xlabel('Generation')
    plt.ylabel('Fitness of the Best Individual')
    # leg1 = plt.legend([ha, plotcollection[0], hblank, hblank, hb, pltphencoll[0], pltphencoll[1], pltphencoll[2]], [cats[0], 'No Swaps', '' ,'', cats[1], '20 Swaps', '100 Swaps', '400 Swaps'] , loc=4, ncol =2)
    # leg1 = plt.legend([ha, ha, ha, hb, hblank, hblank, plotcollection[0], ha, ha, pltphencoll[0], pltphencoll[1], pltphencoll[2]], [cats[0], '','', cats[1], '', '', 'No Swaps','', '','20 Swaps', '100 Swaps', '400 Swaps'], loc = 4, ncol =2)
    leg1 = plt.legend([ha, ha, hb, hblank, plotcollection[0], ha, pltgencoll[0], pltphencoll[0]], [cats[0], '', cats[1],'', 'Genotypic-Fitness','' ,'Genotypic-Fitness', 'Phenotypic-Fitness'], loc=4, ncol=2)
    plt.gca().add_artist(leg1)


    # plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.savefig('./final_figures/Exp1:A', dpi=300)
    plt.show()

def print_stuff(config):
    hw_runs_collection = np.load('./exp1/Populations_hw.npy', allow_pickle=True)
    comp_runs_collection = np.load('./exp1/Populations_comp(genome+phenotype).npy', allow_pickle=True)

    print(hw_runs_collection)




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
            #'REORG_TYPE': 3,
            'ExpFitness': True,
            'BubbleLimit': 0,
            'viewfield': 1,
            'Loops': 10,
        }

    bubbles = np.array([20, 100, 400])
    print('Experiment 1: Running HW versus Competent for {} set of competent levels: {}\n'.format(len(bubbles), bubbles))


    # hw_runs = np.zeros((config['Loops'], config['RUNS']))
    # comp_genome_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))
    # comp_phenotype_runs = np.zeros((len(bubbles), config['Loops'], config['RUNS']))

    # hw_runs_collection = {}
    # comp_runs_collection = {}

    # for yu in tqdm(range(config['Loops'])):
    #     print('**'*10)
    #     print('In Run : {}'.format(yu))
    #     print('**'*10)
    #     print('\n')
    #     print('Hardwired Run...')
    #     fits, hw_genome_populations = run_ga(config, HW=True)
    #     hw_runs[yu, :] = fits
    #     hw_runs_collection[yu] = hw_genome_populations
    #     # comp_runs_collection[yu] = populations_tuple[1]

    #     print('Competent Runs...')
    #     comp_bubbles_collection = {}
    #     for k, r in enumerate(bubbles): 
    #         print('Competency run: {} bubbles'.format(r))
    #         config['BubbleLimit'] = r
    #         cp_gen_fits, cp_cmp_fits, populations_tuple = run_ga(config, HW=False)
    #         comp_genome_runs[k, yu, :], comp_phenotype_runs[k, yu, :] = cp_gen_fits, cp_cmp_fits 
    #         print('Hardwired_Genome: {}'.format(fits[-5:]))
    #         print('Competent_Genome: {}, Competent_Phenotype: {}'.format(cp_gen_fits[-5:], cp_cmp_fits[-5:]))

    #         comp_bubbles_collection[r] = populations_tuple
    #         print('Done')

    #     comp_runs_collection[yu] = comp_bubbles_collection

    #     print('Saving run {}... \n'.format(yu))
    #     np.save('./exp1/Hw_bubblesort', hw_runs)
    #     np.save('./exp1/Comp_genome_bubblesort', comp_genome_runs)
    #     np.save('./exp1/Comp_phenotype_bubblesort', comp_phenotype_runs) 
    #     np.save('./exp1/Populations_hw', hw_runs_collection)
    #     np.save('./exp1/Populations_comp(genome+phenotype)', comp_runs_collection)

    plot_all(bubbles, tar=config['Loops'], max_n=1000)
    # print_stuff(config)