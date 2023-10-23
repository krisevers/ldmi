configfile: "config.yaml"

"""
Workflow for generating sets of parameters for simulations, performing simulations in parallel on clusters,
defining experimental conditions and observables, training and testing neural density estimators, and analysing posterior samples.
"""

# areas = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd', 'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 
#          'LIP', 'PITv', 'PITd', 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST','7a', 'STPp', 'STPa', '46', 'AITd']

rule initiliazation:
    output: "experiments/{wildcards.experiment}/settings.yaml"
    shell: "mkdir -p {{exp_dir}}/{wildcards.experiment} && touch {{exp_dir}}/{wildcards.experiment}/settings.yaml"