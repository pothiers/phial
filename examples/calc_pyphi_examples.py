import sys, inspect
import csv
from pathlib import Path
# External packages
import pyphi
import pyphi.examples as ex
import networkx as nx
# Local packages
from phial.utils import tic,toc,Timer
from phial.experiment import Experiment, timeout_run


funcLUT = dict(
    (name.replace('_func',''),obj)
    for name,obj in inspect.getmembers(ex) if inspect.isfunction(obj))


# Following modified from: list(funcLUT.keys())
subsystems = [
    'PQR',
    'basic_subsystem',
    'basic_noisy_selfloop_subsystem',
    'residue_subsystem',
    'xor_subsystem',
    'macro_subsystem',
    ]
networks = [ # network, state
    # See subsystem 'PQR_network',
    # See subsystem 'basic_network',
    # See subsystem 'basic_noisy_selfloop_network',
    # See subsystem 'residue_network',
    # See subsystem 'xor_network',
    ('propagation_delay_network',   (0,0,0,1,0,0,0,0,0)),  #21 secs, chimp20
    # See subsystem 'macro_network',
    ('blackbox_network', (0,0,0,0,0,0)), 
    ('rule110_network',  (0,0,0)),
    ('rule154_network',  (0,0,0,0,0)),
    ('fig1a',            (0, 0, 0, 0, 0, 0)),
    ('fig3a',            (0, 0, 0, 0)),
    ('fig3b',            (0, 0, 0, 0)),
    ('fig4',             (0, 0, 0)),
    ('fig5a',            (0, 0, 0)),
    ('fig5b',            (1, 0, 0)),
    ('fig16',            (0, 0, 0, 0, 0, 0, 0)),
    ('actual_causation', (0, 0)),
    ('disjunction_conjunction_network', (0, 0, 0, 0)),
    #
    #'all_states',
    # DUPE 'fig10',
    # DUPE 'fig14',
    # DUPE 'fig6',
    # DUPE 'fig8',
    # DUPE 'fig9',
    # 'prevention', # Transistion
]
tpms = [
    'cond_depend_tpm',
    'cond_independ_tpm',
]


def calc_examples(csvfile=None, verbose=False):
    """Calculate Phi for all known pyphi examples.

    Current set comes from ``pyphi.examples``.
    Record: Name of example, State, Phi, compute time (secs)

    :param csvfile: Name of CSV file to write results to. Default: do not write
    :param verbose: Output progress. 
    :returns: dict(results=(name,state,phi,secs), conf=dict(name,val))
    :rtype: dict

    """
    conf =pyphi.config.snapshot()
    results = []
    
    # nnslist:: [(name,network,state), ...]
    nnslist = [(n,funcLUT[n]().network,funcLUT[n]().state) for n in subsystems]
    nnslist += [(n,funcLUT[n](),s) for n,s in networks]
    for name,network,state in nnslist:
        subsys = pyphi.Subsystem(network, state)
        statestr = ''.join(f'{i:x}' for i in state)
        if verbose:
            print(f'COMPUTING: {name}({statestr})', end='')
        tic()
        phi = pyphi.compute.phi(subsys)
        secs = toc()
        if verbose:
            print(f' phi={phi} elapsed_seconds=({secs})')
        results.append((name, state, phi, secs))
        
    if csvfile is not None:
        numcol = len(results[0])
        with open(Path(csvfile).expanduser(), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'State', 'Phi', 'Time (secs)'])
            for row in results:
                writer.writerow(row)
            writer.writerow(['']*numcol)
            writer.writerow(['CONFIGURATION VALUES','','','',''])
            writer.writerow(['CONF KEY','VALUE','','',''])
            for it in conf.items():
                writer.writerow(it)
        print(f'Wrote results to CSV file: {csvfile}')
    return dict(results=results, conf=conf)

def calc_nxgraphs(csvfile=None, verbose=False, timeout=1):
    """Calc phi for random network graphs.
    
    Some of these may take a very long time.  To avoid spending too much time on 
    any one, stop calc if it takes longer than TIMEOUT seconds.  

    https://networkx.github.io/documentation/stable/reference/generators.html

    :param timeout: Number of seconds to let compute run before terminating it.
    :returns: results
    :rtype: dict

    """
    pyphi.config.PARALLEL_CUT_EVALUATION = False
    conf =pyphi.config.snapshot()

    graph_list = [
        # There are 1253 graphs in graph_atlas
        nx.graph_atlas(49),
        nx.balanced_tree(2,2),
        nx.complete_graph(5),
        nx.complete_multipartite_graph(2,3,3),
        nx.circular_ladder_graph(3),
        nx.circulant_graph(5,[1,2]),
        nx.cycle_graph(5),
        nx.dorogovtsev_goltsev_mendes_graph(2),
        #!nx.ladder_graph(3),
        #!nx.margulis_gabber_galil_graph(2),
        #!nx.chordal_cycle_graph(5),
        ]
        
    long_running = []
    computations = {}
    for i,G in enumerate(graph_list):
        exp = Experiment(nx.DiGraph(G).edges, default_func='MJ')

        if G.name:
            gname = f'{G.name}_{i}'
        else:
            gname = G.name or f'G_{i}'        

        print(f'Computing phi for graph: {gname}')
        res = timeout_run(exp, timeout=timeout)

        if res is None:
            long_running.append(gname)
        else:
            computations[gname] = (res.get('phi'), res.get('duration'))
        
    
    return dict(aborted=long_running,
                results=computations,
                timeout=timeout,
                conf=conf)
