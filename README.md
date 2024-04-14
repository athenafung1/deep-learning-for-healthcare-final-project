# CS 598: Deep Learning for Healthcare Final Project

In this final project, we will be exploring CASTER, a framework that predicts drug interactions with
chemical substructure representations.

## Background
Adverse drug-drug interaction (DDI) is the unintended molecular interactions between drugs. It’s a prominent cause of patient morbidities/mortalities, incurring higher costs and risking patient safety. Due to this and the relatively slow rate at which DDIs are identified and reported, there is major interest in predicting whether two drugs will interact, especially during the design process. Thus the ChemicAl SubstrucTurE Representation framework, or CASTER, was introduced to predict DDIs, improving on the weaknesses of prior works. 

There have been 3 main limitations in previous established works:
* DDIs usually result from the reactions of only a few sub-structures of a drug’s entire molecule, but many drug-drug pairs that are not similar in terms of DDI interaction can still have significant overlap on irrelevant substructures. Previous models often generate drug representations using the entire chemical representation, causing learned representations to be potentially biased toward the large, irrelevant substructures and ultimately nullify learned drug similarities and predictions.
* Existing models do not generalize well to unseen drugs and DDIs due to a small labeled dataset. Many models use external, specific biomedical knowledge for an improvement in performance and cannot be generalized for analyzing interactions between any drugs. Especially when considering drugs in an early development phase, we have little to no knowledge about the specific biomedical properties of the drug. In these use cases, CASTER would provide an improvement to the current state.
* Deep models, though performant, have a large number of parameters, making interpretation of the model’s results difficult. The CASTER model can generate results that allow a relatively straightforward interpretation.

## Specific Approach

The CASTER framework has a couple of notable parts: 
* A sequential pattern mining method to efficiently characterize the functional sub-structures of drugs. This accounts for the interaction mechanism between drugs, as interactions depend primarily on the reaction between these functional sub-structures. With an explicit representation of a drug-pair as a set of their functional sub-structures, CASTER can address limitation A.
* An autoencoding module that extracts information from both labeled and unlabeled chemical structure data and embeds patterns into a latent-feature representation that can be generalized to novel drug pairs, addressing limitation B.
* A dictionary learning module to explain the prediction using only a small set of coefficients measuring the relevance of each input sub-structure to the DDI prediction, addressing limitation C. This is done by projecting drug pairs onto the subspace defined by the generalized embeddings generated by the autoencoding module.

## Hypotheses
The authors of CASTER presented the following hypotheses:
* The CASTER model will provide a more accurate DDI when compared with other established models.
* The usage of unlabelled data to generate frequent sub-structure features improves performance in situations with limited labeled datasets.
* CASTER’s sub-structure dictionary can help human operators better comprehend the final result.

## Ablations Planned
We plan to attempt two ablations on the CASTER model:
 
1. Replacement of the deep predictor of the functional representation with a logistic regression.
The molecule’s functional representation (sub-structured/post pattern mining) is fed through the deep predictor. The logistic regression model, though far lighter in parameter count, was not chosen due its weaker performance. We’d like to replicate and verify this by replacing the deep predictor with a logistic regression model.
2. Retraining after the removal/minimization of the unlabelled molecule dataset.
CASTER claims and shows that the inclusion of unlabelled data for better frequent sub-structure generation leads to an improved model performance. To verify this, we would like to restrict the size of / remove the unlabelled dataset from the training process and verify that performance does indeed decrease.

## Data
Since CASTER’s code repository already contains the needed data, we will access it there:
https://github.com/kexinhuang12345/CASTER/tree/master/DDE/data

## Computation Feasibility
According to the paper, the authors used “... a server with 2 Intel Xeon E5-2670v2 2.5GHz CPUs, 128 GB RAM and 3 NVIDIA Tesla K80 GPUs.”. While we don’t have the same amount of memory nor GPUs, this is by no means an unreasonable amount of processing power, considering the speed of more modern hardware. If need arises, we have access to typical cloud compute services, like AWS, GC, or Azure.


### Authors
* [Brian Leung](brianl4@illiois.edu)
* [Gautam Putcha](gputcha2@illinois.edu)
* [Athena Fung](affung2@illinois.edu)
