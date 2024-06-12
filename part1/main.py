""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm
import copy

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """
def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE """
    if A.is_empty():
        return B
    if B.is_empty():
        return A
    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    out.val = np.array(A.val)[idxA] * np.array(B.val)[idxB]
    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE
     HINT: Use the code from lab1 """
    out.var = np.setdiff1d(factor.var, var)
    out.card = factor.card[np.isin(factor.var, out.var)]
    out.val = np.zeros(np.prod(out.card))

    unNormVar=np.squeeze(np.take(factor.get_all_assignments(),np.where(~np.isin(factor.var, var)),axis=1),axis=1)
    unNormVarIndex = assignment_to_index(unNormVar, out.card)

    unNormVal= [np.sum(np.array(factor.val)[unNormVarIndex == i]) for i in range(len(out.val))] 
    out.val=unNormVal/np.sum(unNormVal)
    #out.val=unNormVal
    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    for k in np.intersect1d(out.var,list(evidence.keys())):
            # print("considering: ",k,evidence[k])
            currentVarAssign=np.squeeze(np.take(out.get_all_assignments(),np.where(np.isin(out.var, k )),axis=1),axis=1)
            indexesToChange=np.squeeze(np.where(np.any(currentVarAssign!=evidence[k], axis=1)))
            out.val[indexesToChange]=0.0
    """ END YOUR CODE HERE """

    return out

def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    joint = factors[0]
    if len(factors)<2:return joint

    for factor in factors[1:]:
        joint = factor_product(joint, factor)

    return joint

def update_factor_dict_with_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    holdingDict={}
    for f in out.items():
        factor=f[1]
        node=f[0]
        listToMarginalize=[]
        for k in np.intersect1d(factor.var,list(evidence.keys())):
            # print("considering: ",k,evidence[k])
            currentVarAssign=np.squeeze(np.take(factor.get_all_assignments(),np.where(np.isin(factor.var, k )),axis=1),axis=1)
            indexesToChange=np.squeeze(np.where(np.any(currentVarAssign!=evidence[k], axis=1)))
            factor.val[indexesToChange]=0.0
            #print(factor,k)
            holdingDict[node]=k
        if listToMarginalize:
            holdingDict[node]=factor_marginalize(factor,listToMarginalize)
        else:
            holdingDict[node]=factor
    out=holdingDict
    """ END YOUR CODE HERE """

    return out


""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    nodesToSample=copy.deepcopy(proposal_factors)
    for node in nodes:
        currentFactor=factor_marginalize(factor_evidence(nodesToSample[node],samples),list(samples.keys()))
        # Get the proposal distribution for the node
        proposal_distribution = currentFactor.val
        sample_space = np.arange(len(proposal_distribution))
        # Sample a value using the proposal distribution
        sampled_value = np.random.choice(sample_space, p=proposal_distribution)
        samples[node] = sampled_value
    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples




def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    nodes = list(proposal_factors.keys())
    accumulated_weights = []
    # update evidence
    evidencedTargetFactors=update_factor_dict_with_evidence(target_factors,evidence)
    queryExceptEvidenceVarList=list(set(proposal_factors.keys()).difference(set(evidence.keys())))
    proposalSelectedFactors={k:v for k,v in proposal_factors.items() if k in queryExceptEvidenceVarList}


    for _ in tqdm(range(num_iterations)):
        # Sample using the proposal distribution
        sample = _sample_step(nodes, evidencedTargetFactors)
        pFactors=copy.deepcopy(evidencedTargetFactors)
        pSampledFactors=update_factor_dict_with_evidence(pFactors,sample)
        # compute p(xi)
        pSampleValue=compute_joint_distribution(list(pSampledFactors.values()))
        # sample already contains information from evidence
        # compute q(xi)
        qFactors=copy.deepcopy(proposalSelectedFactors)
        qSampledValues= update_factor_dict_with_evidence(qFactors,sample)
        qVSampleValue=compute_joint_distribution(list(qSampledValues.values()))
        # compute sample weight = p/q
        sampleWeight = Factor(var=pSampleValue.var,card=pSampleValue.card,val=np.divide(pSampleValue.val, qVSampleValue.val, out=np.zeros_like(pSampleValue.val), where=pSampleValue.val!=0))
        accumulated_weights.append(sampleWeight)
    
    #accumulate weights and normalize
    aggWeightsValue=np.sum(np.array([f.val for f in accumulated_weights]),axis=0)
    aggWeightsValue/=np.sum(aggWeightsValue,axis=0)
    normalizedWeights=Factor(var=accumulated_weights[0].var,card=accumulated_weights[0].card,val=aggWeightsValue)
    # sum out provided evidences to reduce bloat
    out=factor_marginalize(normalizedWeights,list(evidence.keys()))

    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
