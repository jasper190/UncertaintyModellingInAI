""" CS5340 Lab 4 Part 2: Gibbs Sampling
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""


import copy
import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from factor_utils import factor_evidence, factor_marginalize, assignment_to_index


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, 'ground-truth')

""" HELPER FUNCTIONS HERE """
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

""" END HELPER FUNCTIONS HERE"""


def _sample_step(nodes, factors, in_samples):
    """
    Performs gibbs sampling for a single iteration. Returns a sample for each node

    Args:
        nodes: numpy array of nodes
        factors: dictionary of factors e.g. factors[x1] returns the local factor for x1
        in_samples: dictionary of input samples (from previous iteration)

    Returns:
        dictionary of output samples where samples[x1] returns the sample for x1.
    """
    samples = copy.deepcopy(in_samples)

    """ YOUR CODE HERE """
    for node in nodes:
        currentVariable=node
        fixedVariablesDict={k:v for k,v in samples.items() if k!=currentVariable}
        #evidence is a problem
        currentFactor=factor_evidence(factors[currentVariable],fixedVariablesDict)
        variable_factor=factor_marginalize(currentFactor,np.setdiff1d(currentFactor.var, [currentVariable]) )
        # print(node,currentFactor.var, [currentVariable],np.setdiff1d(currentFactor.var, [currentVariable]) )
        proposal_distribution = variable_factor.val
        sample_space = np.arange(len(proposal_distribution))
        # Sample a value using the proposal distribution
        sampled_value = np.random.choice(sample_space, p=proposal_distribution)
        samples[node]=sampled_value
        #print(variable_factor,sampled_value,current_samples)
    """ END YOUR CODE HERE """

    return samples


def _get_conditional_probability(nodes, edges, factors, evidence, initial_samples, num_iterations, num_burn_in):
    """
    Returns the conditional probability p(Xf | Xe) where Xe is the set of observed nodes and Xf are the query nodes
    i.e. the unobserved nodes. The conditional probability is approximated using Gibbs sampling.

    Args:
        nodes: numpy array of nodes e.g. [x1, x2, ...].
        edges: numpy array of edges e.g. [i, j] implies that nodes[i] is the parent of nodes[j].
        factors: dictionary of Factors e.g. factors[x1] returns the conditional probability of x1 given all other nodes.
        evidence: dictionary of evidence e.g. evidence[x4] returns the provided evidence for x4.
        initial_samples: dictionary of initial samples to initialize Gibbs sampling.
        num_iterations: number of sampling iterations
        num_burn_in: number of burn-in iterations

    Returns:
        returns Factor of conditional probability.
    """
    assert num_iterations > num_burn_in
    conditional_prob = Factor()

    """ YOUR CODE HERE """
    print(nodes,edges)
    #update evidence
    evidencedFactorsDict=update_factor_dict_with_evidence(factors,evidence)
    evidencedFactorsDict
    # create children dict
    markovBlanketDict= {node:[node]+[child for parent,child in edges if parent==node] for node in nodes}
    singleFactorsDict={}
    # marginalizing to their base prob
    for node in nodes:       
        factorsToMarginalizeOutList=np.setdiff1d(factors[node].var, [node])
        # print(node,markovBlanketDict[node])
        singleFactorsDict[node]=factor_marginalize(factors[node],factorsToMarginalizeOutList)
        #singleFactorsDict[node]=currentNodeFactor
        print(singleFactorsDict[node])
    
    nodes=[n for n in nodes if n not in evidence.keys()]

    markovSeperatedDict={}
    for node in nodes:
        markovJoinedVariablesList=markovBlanketDict[node] 
        print(node,markovJoinedVariablesList)
        currentMarkovFactor=compute_joint_distribution([evidencedFactorsDict[n] for n in markovJoinedVariablesList])
        factorsToMarginalizeOutList=np.setdiff1d(currentMarkovFactor.var, markovJoinedVariablesList)
        markovSeperatedDict[node]=factor_marginalize(currentMarkovFactor,factorsToMarginalizeOutList)
        print(markovSeperatedDict[node])
       
    # removing evidence from samping
    start_samples={k:v for k,v in copy.deepcopy(initial_samples).items() if k not in evidence.keys()}

    burnedInSamples={}
    # num_iterations, num_burn_in
    for _ in tqdm(range(num_burn_in)):
        burnedInSamples=_sample_step(nodes=nodes, factors=singleFactorsDict, in_samples=start_samples)

    samplesList=[]
    for _ in tqdm(range(num_iterations-num_burn_in)):
        samplesList.append(_sample_step(nodes=nodes, factors=singleFactorsDict, in_samples=burnedInSamples))
    
    #for sample in samplesList:
    #    for k,v in evidence.items():
    #        sample[k]=v

    df1=pd.DataFrame(samplesList)
    grouped = df1.groupby(list(df1.columns)).size().reset_index(name='counts')
    problemVal=np.zeros(len(node_factors[0].val))
    problemVal[grouped.apply(lambda x: assignment_to_index(x[df1.columns].to_numpy(),node_factors[0].card[nodes]),axis=1)]=grouped.counts

    conditional_prob=Factor(var=nodes,card=factors[0].card[nodes],val=problemVal/np.sum(problemVal))
    """ END YOUR CODE HERE """

    return conditional_prob


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
    proposal_factors_dict = input_config['proposal-factors']

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    nodes = np.array(input_config['nodes'], dtype=int)
    edges = np.array(input_config['edges'], dtype=int)
    node_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                    node, proposal_factor_dict in proposal_factors_dict.items()}

    evidence = {int(node): ev for node, ev in input_config['evidence'].items()}
    initial_samples = {int(node): initial for node, initial in input_config['initial-samples'].items()}

    num_iterations = input_config['num-iterations']
    num_burn_in = input_config['num-burn-in']
    return nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in


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
    nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in = \
        load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(nodes=nodes, edges=edges, factors=node_factors,
                                                           evidence=evidence, initial_samples=initial_samples,
                                                           num_iterations=num_iterations, num_burn_in=num_burn_in)
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
