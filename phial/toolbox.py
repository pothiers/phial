# Python standard library
from collections import Counter, defaultdict
import itertools
from random import choice
import subprocess
import json
import re
import operator
import functools
# External packages
import networkx as nx
from networkx.drawing.nx_pydot import write_dot,pydot_layout
import pandas as pd
import numpy as np
import pyphi
import pyphi.network
from pyphi.convert import sbn2sbs, sbs2sbn, to_2d
from IPython.display import Image
# Local packages
import phial.node_functions as nf

def rep_nth_char(ss, n, c=''):
    """Remove or replace nth char in string SS"""
    return ss[:n] + c + ss[n+1:]


def nodes_state(state, nodelabels):
    """Convert system state (statehexstr) to dict[nodeLabel]=nodeState"""
    return dict((n,int(s,16)) for n,s in zip(nodelabels, state))

def system_state(nodes_state):
    """Convert nodes_state (dict[label]=state) to statehexstr"""
    return ''.join(f'{i:x}' for i in nodes_state.values())
    
def all_states(N, spn=2, backwards=False):
    """All combinations spn^N binary states in lexigraphical order.
    This is NOT the order used in most IIT papers.  
    To get order for papers, set 'backwards=True'.
    RETURN list of statestr in hex format. e.g. '01100'
    spn:: States Per Node
    """
    assert spn <= 16 # because we represent as hex string.
    states = [''.join(f'{spn:x}' for spn in sv)
              for sv in itertools.product(range(spn),repeat=N)]
    if backwards:
        states = sorted(states, key= lambda s: s[::-1])
    return states

def dotgraph(G, pngfile):
    """Return networkx DiGraph. Maybe write to PNG file."""
    dotfile = pngfile + ".dot"
    write_dot(G, dotfile)
    cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
    with open(pngfile,'w') as f:
        subprocess.check_output(cmd, shell=True)
    return Image(filename=pngfile)
    
# NB: This does NOT hold the state of a node.  That would increase load
# on processing multiple states -- each with its own set of nodes!
# Instead, a statestr contains states for all nodes a specific time.
#
# Would this be light-weight enough 10^5++ nodes? @@@
#
# InstanceVars: id, label, num_states, func
class Node():
    """Node in network. Supports more than just two states but downstream 
    software may be built for only binary nodes. Auto increment node id that 
    will be used as label if one isn't provided on creation.
    """
    _id = 0

    def __init__(self,label=None, num_states=2, id=None, func='MJ'):
        if id is None:
            id = Node._id
            Node._id += 1
        self.id = id
        self.label = label or id
        self.num_states = num_states
        self.func = func 
        
    def truth_table(self, max_inputs=4):
        """Full truth table for function associated with Node. Inputs consist
        of all possible lists of binary values up to length 'max_inputs'.
        """
        table = []
        for length in range(max_inputs+1):
            table.extend([(''.join(str(s) for s in sv),self.func(sv))
                          for sv in itertools.product([0,1],repeat=length)])
        return table

    @property
    def random_state(self):
        return choice(range(self.num_states))

    @property
    def states(self):
        """States supported by this node."""
        return range(self.num_states)

    #!def __repr__(self):
    #!    return ('Node('
    #!            f'label={self.label}, '
    #!            f'id={self.id}, '
    #!            f'num_states={self.num_states}, '
    #!            f'func={self.func.__name__}')
    #!
    #!def __str__(SELF):
    #!    return f'{self.label}({self.id}): {self.num_states},{self.func.__name__}'

    
class Net():
    """Store everything needed to calculate phi.
    InstanceVars: graph, node_lut, tpm

    :param edges: connectivity edges; e.g. [(0,1), (1,2), (2,0)]
    :param tpm: default: use node funcs to calc
    :param N: Number of nodes
    :param SpN: States per Node
    :param title: Label for connectivity graph
    :param func: default function (mechanism) for all nodes

    """

    _nn = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    
    def __init__(self,
                 edges = None, # connectivity edges; e.g. [(0,1), (1,2), (2,0)]
                 tpm = None, # default: using node funcs to calc
                 N = 5, # Number of nodes
                 #graph = None, # networkx graph
                 SpN = 2,  # States per Node
                 title = None, # Label for graph
                 func = 'MJ', # default mechanism for all nodes
                 ):
        G = nx.DiGraph()
        if edges is None:
            n_list = range(N)
        else:
            i,j = zip(*edges)
            maxid = max(i+j)
            n_list = sorted(set(i+j))
        edgesStrP = type(n_list[0]) == str
        if edgesStrP:
            n_list = [(ord(l) - ord('A')) for l in n_list]
        nodes = [Node(id=i, label=Net._nn[i], num_states=SpN, func=func)
                 for i in n_list]

        # lut[label] -> Node
        self.node_lut = dict((n.label,n) for n in nodes)
        #invlut[i] -> label
        invlut = dict(((n.id,n.label) for n in self.node_lut.values())) 
            
        G.add_nodes_from(self.node_lut.keys())
        if edges is not None:
            if edgesStrP:
                G.add_edges_from(edges)
            else:
                G.add_edges_from([(invlut[i],invlut[j]) for (i,j) in edges])
        self.graph = G
        self.graph.name = title
        if tpm is None:
            self.tpm = self.calc_tpm()
        else:
            allstates = all_states(len(self.graph), backwards=True)
            allnodes = [n.label for n in nodes]
            self.tpm = pd.DataFrame(tpm, index=allstates, columns=allnodes)
            self.tpm.index.name = ''.join([n.label for n in self.nodes])

    #!!!
    def nodeTpm(self, nodeLabel):
        """Generate local TPM for node from system TPM.
        Resulting TPM only uses in-states that matter to the result of nodeLabel.
        Accepts non-binary nodes.
        """
        # Try all nodes in states to see if each matters for out-state of
        # nodeLabel truth-table.  If node matters for any rows,
        # keep it, else remove it from all states and
        # remove duplicate state entries.
        
        
        def state_node_variants(state,node):
            """Return copies of STATE modified with all possible states of NODE
            """
            return [rep_nth_char(state,node.id,str(ns))
                    for ns in range(node.num_states)]

        tt = self.tpm.loc[:,nodeLabel] # full truthTable for NODE
        
        # Find input nodes that matter (causes)
        causes = set() # Nodes that effect truth table
        for node in self.nodes:
            for state in tt.index:
                snv = state_node_variants(state,node)
                # NODE makes a difference to out-states for nodeLabel
                if not functools.reduce(operator.eq, [tt.loc[s] for s in snv]):
                    causes.add(node)
                   
        # Remove columns that don't matter to truthTable
        cause_ids = [n.id for n in causes]
        nuke_ids = set(range(len(self))).difference(cause_ids)
        #! print(f'cause nodes={[n.label for n in causes]}')
        #! print(f'nuke_ids={nuke_ids}')
        new_tt_lut = dict() # d[localState] = nodeState
        for state in tt.index:
            s = state
            for i in sorted(nuke_ids,reverse=True):
                s = rep_nth_char(s,i)
            new_tt_lut[s] = tt.loc[state]

        df = pd.DataFrame(new_tt_lut.values(),
                          index=new_tt_lut.keys(), columns=[nodeLabel])
        df.index.name = ''.join(sorted([n.label for n in causes]))
        return df
    
    @property
    def state_graph(self):
        """Get the state-to-state graph labeled with states.

        :returns: state-to-state graph
        :rtype: networkx.DiGraph

        """
        G = nx.DiGraph(sbn2sbs(self.tpm))
        mapping = dict(zip(range(len(self.tpm.index)), self.tpm.index))
        S = nx.relabel_nodes(G, mapping)
        return S

    def from_json(self, jsonstr,
                  func = 'MJ', # default mechanism for all nodes
                  SpN=2):
        """Overwrite contents of this Net with data from jsonstr.

        :param jsonstr: JSON format of Net data
        :param func: default node function to attach to all nodes in net.
        :param SpN: default number of States per Node

        :returns: TPM
        :rtype: pandas.DataFrame

        """
        self.graph = self.node_lut = self.tpm = None
        jdict = json.loads(jsonstr)
        edges = jdict.get('edges',[])
        i,j = zip(*edges)
        n_list = sorted(set(i+j))
        nodes = [Node(**nd) for nd in jdict.get('nodes',[])]
        for n in nodes:
            n.func = nf.funcLUT[n.func]
        
        self.graph = nx.DiGraph(edges)
        self.node_lut = dict((n.label,n) for n in nodes)

        S = nx.DiGraph(jdict.get('tpm',[]))
        #self.tpm = sbs2sbn(nx.to_pandas_adjacency(S))
        s2s_df = nx.to_pandas_adjacency(S)
        df = pd.DataFrame(index=s2s_df.index, columns=self.node_lut.keys())
        for istate in s2s_df.index:
            for outstate in s2s_df.columns:
                if s2s_df.loc[istate,outstate] == 0:
                    continue
                ns = nodes_state(outstate,self.node_lut.keys())
                print(f'DBG: istate={istate} ns={ns}')
                df.loc[istate] = list(ns.values())
        self.tpm = df

    def to_json(self, filename=None):
        """Output contents of this Net in JSON format.

        :param filename: where to write JSON (or do not write)
        :returns: python dictionary matching the JSON
        :rtype: dict

        """
        S = self.state_graph
        jj = dict(
            edges=list(self.graph.edges),
            tpm = list(S.edges),
            nodes=[dict(label=n.label,
                        id=n.id,
                        #func=re.sub('_func$','',n.func.__name__),
                        num_states=n.num_states )
                   for n in self.nodes],
        )
        if filename is not None:
            with open(filename, 'w') as f:
                json.dump(jj, fp=f)
        return jj

    def info(self):
        """Get information this Net

        :returns: dict containing: edges,nodes, various counts
        :rtype: dict

        """
        dd = dict(
            edges=list(self.graph.edges),
            nodes=[str(n) for n in self.nodes],
            num_in_states=len(self.in_states),
            num_unreachable_states=len(self.unreachable_states),
            num_state_cc = self.state_cc,
            num_state_cycles = len(list(self.state_cycles))
        )
        return dd
        
    @property
    def state_cc(self):
        """Number of connected components in state-to-state graph (TPM)."""
        S = nx.DiGraph(sbn2sbs(self.tpm))
        return nx.number_weakly_connected_components(S)

    @property
    def state_cycles(self):
        """Cycles found in state-to-state graph. (TPM)"""
        S = nx.DiGraph(sbn2sbs(self.tpm))
        return nx.simple_cycles(S)
        
    def node_state_counts(self, node):
        """Truth table of node.func run over all possible inputs.
        Inputs are predecessor nodes with all possible states."""
        preds = (self.get_node(l)
                 for l in set(self.graph.predecessors(node.label)))
        counter = Counter()
        counter.update(node.func(sv)
                       for sv in itertools.product(*[n.states for n in preds]))
        return counter

    def eval_node(self, node, system_state_tup):
        preds_id = set([self.get_node(l).id
                    for l in set(self.graph.predecessors(node.label))])
        args = [system_state_tup[i] for i in preds_id]
        return node.func(args) # args are in node order
        

    def node_pd(self, node):
        """Probability Distribution of NODE states given all possible inputs
        constrained by graph."""
        #node = self.get_node(node_label)
        counts = self.node_state_counts(node)
        total = sum(counts.values())
        return [counts[i]/total for i in node.states]

    def calc_tpm(self):
        """Iterate over all possible input states. 

        Use node funcs
        to calculate output state. State-to-State form. Allows non-binary.
        Does not save the resulting TPM.

        :returns: system TPM
        :rtype: pandas.DataFrame

        """
        backwards=True  # I dislike the order the papers use!
        allstates = list(itertools.product(*[n.states for n in self.nodes]))
        N = len(self.nodes)
        allstatesstr = [''.join([f'{s:x}' for s in sv]) for sv in allstates]
        df = pd.DataFrame(index=allstatesstr,
                          columns=[n.label for n in self.nodes]).fillna(0)
        
        for sv in allstates:
            s0 = ''.join(f'{s:x}' for s in sv)
            for i in range(N):
                node = self.nodes[i]
                nodestate = self.eval_node(node,sv)
                df.loc[s0,node.label] =  nodestate

        if backwards:
            newindex= sorted(df.index, key= lambda lab: lab[::-1])
            return df.reindex(index=newindex)
        return df.astype(int)

    @property
    def out_states(self):
        """Output states of TPM in hexstr form. These are the states allowed
        for the 'statestr' phi method.
        Otherwise the error 'cannot be reached in the given TPM' is thrown."""
        return set(''.join(f'{int(s):x}' for s in self.tpm.iloc[i])
                   for i in range(self.tpm.shape[0]))
    @property
    def in_states(self):
        """All possible states of the system.
        Order matches what is commonly used in IIT papers. 
        This is NOT lexicographical order.
        A state is given as a string of hex digits ordered to match
        the order of nodes in the Net.

        :returns: list of hexstrings
        :rtype: pandas.Index

        """
        return self.tpm.index

        """System states that are not reachable from any input states."""

    @property
    def unreachable_states(self):
        """Input states that are not also output states.

        :returns: list of hexstrings
        :rtype: pandas.Index

        """
        return sorted(set(self.in_states) - self.out_states)
        

    @property
    def cm(self):
        """Get Connectivity Matrix
        
        see also: cm_df

        :returns: CM in form suitable for pyphi.Network()
        :rtype: numpy.array

        """
        return nx.to_numpy_array(self.graph)

    @property
    def cm_df(self):
        """Get Connectivity Matrix (DataFrame)
        
        Labels rows and columns with node labels. 
        See also: cm

        :returns: CM in form that displays well in Notebook
        :rtype: pandas.DataFrame

        """

        return nx.to_pandas_adjacency(self.graph, nodelist=self.node_labels)

    @property
    def nodes(self):
        """Return list of all nodes in ID order."""
        return sorted(self.node_lut.values(), key=lambda n: n.id)

    @property
    def node_labels(self):
        return [n.label for n in self.node_lut.values()]

    def __Xsuccessors(self, node_label):
        return list(self.graph.neighbors(node_label))

    def get_node(self, node_label):
        """Get a Node by its label.

        :param node_label: (str) Label (which shows in Draw) of node to retrieve 
        :returns: instance of Node()
        :rtype: Node

        """
        return self.node_lut[node_label]

    def get_nodes(self, node_labels):
        """Get list of Node instances by list of labels.

        :param node_labels: (str) list of strings
        :returns: list of Node() instances
        :rtype: list

        """

        return [self.node_lut[label] for label in node_labels]
    
    def __len__(self):
        return len(self.graph)

    def gvgraph(self, pngfile=None):
        """Return networkx DiGraph. Maybe write to PNG file."""
        G = nx.DiGraph(self.graph)
        if pngfile is not None:
            dotfile = pngfile + ".dot"
            write_dot(G, dotfile)
            cmd = (f'dot -Tpng -o{pngfile} {dotfile} ')
            with open(pngfile,'w') as f:
                subprocess.check_output(cmd, shell=True)
        return G

    def draw(self):
        """Draw the node connectivity graph (in notebook).

        In ipython, execute ``matplotlib.pyplot.show()`` to force display.

        :returns: net graph
        :rtype: nx.DiGraph

        """
        nx.draw(self.graph,
                pos=pydot_layout(self.graph),
                # label='gnp_random_graph({N},{p})',
                with_labels=True )
        return self.graph

    def draw_states(self):
        """Draw state-to-state graph (in notebook).

        In ipython, execute ``matplotlib.pyplot.show()`` to force display.

        :returns: state graph
        :rtype: nx.DiGraph

        """
        G = nx.DiGraph(sbn2sbs(self.tpm))
        mapping = dict(zip(range(len(self.tpm.index)), self.tpm.index))
        S = nx.relabel_nodes(G, mapping)
        nx.draw(S, pos=pydot_layout(S), with_labels=True)
        return S
            
    @property
    def pyphi_network(self):
        """Return pyphi Network() instance."""
        return pyphi.network.Network(self.tpm.to_numpy(),
                                     cm=self.cm,
                                     node_labels=self.node_labels)

    
    def phi(self, statestr=None, verbose=False):
        """Calculate phi for net."""
        if statestr is None:
            instatestr = choice(self.tpm.index)
            statestr = ''.join(f'{int(s):x}' for s in self.tpm.loc[instatestr,:])
        #!print(f'DBG statestr={statestr}')
        state = [int(c) for c in list(statestr)] 
        if verbose:
            print(f'Calculating Φ at state={state}')
        node_indices = tuple(range(len(self.graph)))
        subsystem = pyphi.Subsystem(self.pyphi_network, state, node_indices)
        return pyphi.compute.phi(subsystem)
#END Net()

def phi_all_states(net):
    """Run pyphi.compute.phi over all reachable states in net."""
    results = dict() # d[state] => phi
    for statestr in net.out_states:
        results[statestr] = net.phi(statestr)
        print(f"Φ = {results[statestr]} using state={statestr}")
    return results

def pyphi_network_to_net(network):
    """Load pyphi network into this instance (overwrite existing data)."""
    labelLUT = dict(zip(network._node_indices, network._node_labels))
    cm = network.cm
    G = nx.DiGraph(cm)
    tpm = to_2d(network.tpm) # sbn form
    net = Net(G.edges)
    allstates = list(itertools.product(*[n.states for n in net.nodes]))
    for si,sv in enumerate(allstates):
        s0 = ''.join(f'{s:x}' for s in sv)
        for ni in range(len(net)):
            node = net.nodes[ni]
            net.tpm.loc[s0,node.label] = tpm[si][ni]
    return net

