#!/usr/env/bin python3


# Functions to run every experiment. 
# Create an instance of this module by passing a hyperparameters file. 

import math
import json
import numpy as np


class HelperFuncs():
  def __init__(self, config):

    self.config = config
    self.ncr = math.factorial(self.config['SIZE']) / (math.factorial(self.config['SIZE']-2)* 2) # calculates the number of possible ways of selecting two different numbers from an array of specific size. All that's being computed is NC2.

  def fitness(self, o):     
    # Calculates the fitness of a single individual/array
    # Input: A single array of shape (,SIZE)
    # returns: the normalized fitness and the number of swaps required to reach maximum fitness

    # you basically check how many array elements are already ordered and return that as a normalized count. If most elements are ordered then fitness would be 1.0 (maximum)

    non_inv_count = 0
    for i in range(len(o)):
      for j in range(i + 1, len(o)):
        if (o[i] <= o[j]):  #<= is chosen because the elements of array are not unique, repeats exist.
          non_inv_count += 1

    x_val = non_inv_count/self.ncr
    if bool(self.config['ExpFitness']) == True:  # an additional option to report exponential fitness / normal fitness
      return (9**x_val) / 9, self.ncr-non_inv_count
    else:
      return x_val, self.ncr-non_inv_count


  def check_feasibility(self, to_swap_array, index):
    # Function to check if the post-swapped array has a greater fitness than the pre-swapped array. Used by function "bubble_sort"
    # Input: the array, the index to swap
    # Returns: the post-swapped array if it has a greater fitness, else a dummy value of [0]

    per = to_swap_array.copy()
    temp = per[index]
    per[index] = per[index+1]
    per[index+1] = temp

    if self.fitness(per)[0] > self.fitness(to_swap_array)[0]:
      return per, True
    else:
      return [0], False

  def bubble_sort(self, orgs):
    # Function to carry out restricted bubble sort
    # Input: a population
    # Returns: the population after individuals undergo swapping

    individuals = orgs.copy()  #.copy() used to avoid array overwriting (numpy)

    for n, _ in enumerate(individuals):

      _, total_swaps = self.fitness(individuals[n]) # gets the fitness of an array + the number of available swaps

      defecit_swaps = total_swaps - self.config['BubbleLimit']  # check if there are enough swaps availabe to swap 

      if defecit_swaps >0:  #if there are, then proceed to swap as many times as the user provides through the "Bubble Limit" hyperparameter
        swapsToUse = self.config['BubbleLimit']
      else:
        #if not, then just swap as many times as the number of swaps available
        swapsToUse = total_swaps

      counter = 0 #keeps track of the number of swaps executed
      notswapped = 0# counter to check when to quit, if every position is checked and no swaps are possible then we exit the while loop

      while counter < swapsToUse and notswapped < self.config['SIZE']-1:
        notswapped = 0  #zeroed at every iteration because to check only over the size of the array

        poses_touse = np.random.choice(np.arange(self.config['SIZE']-1), self.config['SIZE']-1 , replace = False) # randomly pick a position to swap. Basically instead of sequentially moving from one left to right, just pick random positions (uniquely)

        for pos in poses_touse: 
          #start swapping
          if (individuals[n][pos] > individuals[n][pos+1]):
            # if a value at pos x is greater than value at pos x+1 then there is an opportunity to swap
            mod_ind, feasible = self.check_feasibility(individuals[n], pos) #swap and check if the swap increases fitness
            if feasible:
              #if it does, rewrite the array with the swapped version of it
              individuals[n] = mod_ind
              counter +=1 #increment the swap counter
            else:
              notswapped +=1

          else:
            # update the not swapped counter if 1. a swap is not possible(here) and 2. a swap leads to lower fitness (else statement above)
            notswapped +=1

          if (counter >=swapsToUse) or (notswapped >= self.config['SIZE']-1):
            #while swapping, if any counter is at it's limit, break out of the for loop, this will also trigger a break in the while loop
            break

    return individuals 


  
  def bubble_sortevolve(self, orgs):
    # Identical to the normal bubble_sort function, except that each individual has a competency gene value at the last index position

    individuals = orgs.copy()
    genes = [i[-1] for i in individuals] #get all competency genes of the population
    individuals = [i[:-1] for i in individuals] #get individuals without it's competency gene

    for n, _ in enumerate(individuals):
      counter = 0
      _, total_swaps = self.fitness(individuals[n])

      defecit_swaps = total_swaps - genes[n] #the deficit is now calculated by subtracting the competency gene value of each individual

      if defecit_swaps >0:
        swapsToUse = genes[n]
      else:
        swapsToUse = total_swaps

      notswapped = 0

      while counter < swapsToUse and notswapped < self.config['SIZE']-1:
        notswapped = 0  

        poses_touse = np.random.choice(np.arange(self.config['SIZE']-1), self.config['SIZE']-1 , replace = False) 

        for pos in poses_touse: 
          #start swapping
          if (individuals[n][pos] > individuals[n][pos+1]):

            mod_ind, feasible = self.check_feasibility(individuals[n], pos) 

            if feasible:

              individuals[n] = mod_ind
              counter +=1 

            else:
              notswapped +=1

          else:
            notswapped +=1

          if (counter >=swapsToUse) or (notswapped >= self.config['SIZE']-1):
            break

    individuals = np.append(individuals, np.array(genes).reshape(-1,1), axis = 1) #append the respective genes back to the swapped arrays

    return individuals 


  def selection(self, ory, fi, noevolution=False):
    # Function to carry out selection
    # Inputs: the population, and the fitnesses to base the selection on (Phenotypic fitness in our case)
    # Returns: Genomes of the individuals with the best phenotypic fitness (top 10 %)

    organisms = ory.copy()
    fitne = fi.copy()

    fit_index = {k: i for k, i in enumerate(fitne)} # get a mapping of the position of the individual in the population to it's fitness value
    fitness_organisms = {k: v for k, v in sorted(fit_index.items(), key=lambda item: item[1], reverse=True)}  #sort the fitnesees in descending order and get a mapping of their values to their position in the population
    orgs_keys = [k for k,v in fitness_organisms.items()]  #get position of the individuals in the population from the above sorted dict
    orgs_vals = list(fitness_organisms.values()) # get the corresponding fitness values 

    new_orgs_keys = orgs_keys[: round(0.1*self.config['N_organisms'])] #get the top10 individual positions 
    new_orgs_vals = orgs_vals[: round(0.1*self.config['N_organisms'])] #get the top10 individual fitnesses

    if noevolution: #ignore, this is a test condition
      new_orgs_keys = orgs_keys[:]
      new_orgs_vals = orgs_vals[:]

    new_orgs = [organisms[j] for j in new_orgs_keys] # get the genomes of the fittest individuals, whose position in the population we known
    max_fitness = new_orgs_vals[0]  # get the maximum fitness of the population

    return new_orgs, max_fitness  #return the top 10% of fittest individuals and the max fitness as well


  def combined_selection(self, ory):
    # Function to select the best individuals from a MIXED population (containing both hardwired and competent individuals)
    # Inputs: The mixed population 

    organisms = ory.copy()
    pseudo_organisms = np.array([i[1:] if i[0] == -2 else self.bubble_sort(i[1:].reshape(1,-1)).reshape(-1) for i in organisms]) # HW individuals have an indentifier of "-2" and Competent individuals an indentifier of "-1" at index position 0, we only reorganize the competent individuals and store them in an array. This is just done to get the phenotypic fitness of competent and the genomic fitness of HW.

    fitnessess = [self.fitness(j)[0] for j in pseudo_organisms] # Calulate the fitness of each individual in the array created above

    return(self.selection(ory, fitnessess)) # Carry out normal selection and return the top10% of the best individuals. 

  def crossover_mutation(self, organis):
    # Function to reproduce individuals in a population. After selection, population size is reduced. This function mimics reproduction to repopulate the population.
    # Inputs: The population to repopulate
    # Returns: The repopulated population

    organisms = organis.copy()

    L = len(organisms)  # Get the current_size of the population
    while (len(organisms) < self.config['N_organisms']):  # Quit only when the current_size exceeds the number of organisms (N_organisms) allowed.
      i = np.random.randint(L)  # Get the position of a random individual (lets say the male)
      j = np.random.randint(L)  # Get another individual's position (lets say the female)
      if i!=j: # ensure that we don't select the same indexes
        random_pair = (i, j) # group the pair of individuals involved in reproduction
      else:
        continue

      # An illustration of the Cross mutation process is provided in the paper (Fig.1, Panel C). Consulting it will help better understand how we get a new individual from two existing individual.
      cross_over_point = np.random.randint(low = self.config['LOW']+10, high=self.config['SIZE']) # Find the index position to cross mutate
      new_org_1 = organisms[random_pair[0]].copy() # Get the male
      new_org_2 = organisms[random_pair[1]].copy() # Get the female

      temp = new_org_1[:cross_over_point].copy() # Get a 1st part of the male
      new_org_1[:cross_over_point] = new_org_2[:cross_over_point] # Combine the 1st part of the male with the corresponding part of the female
      new_org_2[:cross_over_point] = temp  # Combiine the 1st part of the female with the original 1st part of the male

      organisms = np.append(organisms, [new_org_1, new_org_2], axis =0)  # Add these two new individuals to the population

    return organisms


  def mutation_flip(self, organi):
    # Function to introduce pointwise mutations to individuals of a population.
    # Inputs: The population to mutate.
    # Returns: The mutated population.

    organisms = organi.copy()

    for i in range(len(organisms)):  
      if np.random.rand(1) > self.config['Mutation_Probability']:  # Mutate only if above a certain probability
        organisms[i][np.random.randint(low = 0, high =self.config['SIZE'])] = np.random.randint(low = self.config['LOW'], high=self.config['HIGH']) # Point Mutate to a value between 0 and array_size 

    return organisms     


  def mutation_flip_incl_gene(self, organi, max_val_to_mutate):
    # Function to introduce pointwise mutations, when a functional gene is present (eg: competency gene)
    # Inputs: The population to mutate, the maximum value to mutate a FUNCTIONAL gene to.

    organisms = organi.copy()

    for i in range(len(organisms)):
      if np.random.rand(1) > self.config['Mutation_Probability']:  # Mutate once in a while only
        mut_pos = np.random.randint(low = 0, high= self.config['SIZE']+1) # Get the index position to mutate
        if mut_pos == 50:
          subst_val = np.random.randint(low = self.config['LOW'], high=max_val_to_mutate) # If the index belongs to the funcitonal gene, then mutate between 0 and "max_val_to_mutate"
        else:
          subst_val = np.random.randint(low = self.config['LOW'], high=self.config['HIGH']) # If the index belongs to a structural gene, then mutate between 0 and array_size

        organisms[i][mut_pos] = subst_val # carry out the point mutation

    return organisms         



if __name__=='__main__':
  funcs = HelperFuncs('normal_population')
  # Test block