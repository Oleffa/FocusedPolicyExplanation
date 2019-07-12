# Focused Policy Explanation Generation Framework

## Abstract
Transparency of robot behaviors increases effi ciency and quality of interactions with humans.
To increase transparency of robot policies, we propose a method for generating robust and focused explanations that express why a robot chose a particular action. The proposed method examines the policy based on the state space in which an action was chosen and describes it in natural language.
The method can generate focused explanations by leaving out irrelevant state dimensions, and avoid explanations that are sensitive to small perturbations or have ambiguous natural language concepts.
Furthermore, the method is agnostic to the policy representation and only requires the policy to be evaluated at different samples of the state space.
We conducted a user study with 18 participants to investigate the usability of the proposed method compared to a comprehensive method that generates explanations using all dimensions.
We observed how focused explanations helped the subjects more reliably detect the irrelevant dimensions of the explained system and how preferences regarding explanation styles and their expected characteristics greatly differ among the participants.

## Usage
### Demo
A minimal example is provided in the file `example.py`.
The minimal example generates plots for a synthetic 2D-state space, similar to the example in the paper.
The user has to set the plots_directory and classifier_directory variables in which the plots and dimension specific classifiers will be stored.
### Experiment
The code for the experiment conducted in the paper is in the folder experiment\.
It contains the GUI application that uses the policy explanation framework as well as a demo for the state and action space used in the experiment.

## TODO
- experiment folder
- remove all the references to concepts
- commenting
- clean up plotting
- todos in demo

To cite this work please refer to the full paper published in RO-MAN 2019:

```
@article{focusedpolicyexplanation,
title={},
author:{Struckmeier, Oliver and Racca, Mattia and Kyrki, Ville},
year={2019}
}
```
