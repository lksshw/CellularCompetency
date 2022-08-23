## Cellular Competency

This repository contains experimental simulations from the paper: "Competency of the Developmental Layer Alters Evolutionary Dynamics in an Artificial Embryogeny Model of Morphogenesis". 
Each experiment is setup to probe the disparity between genotypes and phenotypes as a "competent population" undergoes evolution: The genetic code which instructs development is NOT what is eventually developed. Evolution skews and redirects development to an optimum different from that defined by genetic code. Our paper seeks to: 1. Check if such a disparity actually occurs in simulation, and 2. What the nature of such a disparity is. 

A "competent population" is one wherein the cells of an individual have the capability to solve tasks in their local environment. In contrast, an "incompetent" or "Hardwired population" (what we refer to in the paper) is one where cell's of an individual have no local task-solving capability.

In Experiment 1, we pit the "hardwired" population against the "competent" population to check if the Competent population perform better.

In Experiment 2, we mix hardwired and competent individuals together into a single populaiton and evolve them together. We check to see if and when the competents dominate the hardwired. 

In Experiment 3, we make competency evolvable. Previous experiments involved a fixed level of competency; now, we let evolution decide the magnitude of competency. We check to see the effects competency has on the genome and make observations about its nature.

## Requirements

* Numpy
* Matplotlib
* Seaborn

Nothing fancy.

## Simulating

You can either run simulations with our settings or you can use your own. The "settings" I'm refering to are hyperparameter values; all of which are specified in the <a href='../hyperparameters.json'>Hyperparameters file</a>.
<br>
Well then, to simulate Experiment 1, you run:

```
python3 all_tests/Exp1-CompetentVsHw.py --simulate True
```

similarly for Experiment 2:

```
python3 all_tests/Exp2-MixedPrevalence.py --simulate True
```

Experiment 2 comes in two flavors. The default is the one above. The other involves mixing a very small number (20%) of competent with a lot of hardwired (80%) and checking how different comeptencies behave. To run this version of Experiment 2, run:

```
python3 all_tests/Exp2-MixedPrevalence.py --simulate True --minorityExp True 
```

To simulate Experiment 3, run:

```
python3 all_tests/Exp3-EvolvableCompetency.py --simulate True --plotType "fitness"   
```

Assuming you've pre-saved simulation data, you can also plot the frequency of changes between the structural and competency gene using:

```
python3 all_tests/Exp3-EvolvableCompetency.py --plotType "frequency"   
```

### Reproduction

We provide pre-saved simulation runs for each experiment so that you can reproduce all our figures.

Yet to be uploaded ... 
