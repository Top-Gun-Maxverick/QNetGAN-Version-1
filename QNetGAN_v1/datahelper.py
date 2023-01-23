from torch_geometric import *
from torch_geometric.datasets import QM9
from torch_geometric.data import *
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import scipy
import scipy.io
import numpy as np

class DataProcessor(object):
    def __init__(self):
        '''
        class for processing data that comes out of the data loader
        this doesn't need to be in a class but like who asked
        '''
        self.name = "Data Processor"

    def pyg_to_ssm(self, sample):
        '''
        converts a pytorch_geometric molecule to a scipy sparse matrix
        @params
        sample: pytorch_geometric.data --> the molecule to be converted
        '''
        sample = to_networkx(sample)
        sample_mat = nx.to_scipy_sparse_matrix(sample)
        return sample, sample_mat

    def molecule_is_own_MST(self, graph):
        '''
        function that checks whether a molecule is it's own Minimum Spanning Tree if the molecule is it's own MST then it can't be used since the input to the Generator cannot be empty.
        Well, it technically can but there's no point training it on nothing, is there???
        '''
        graph = nx.to_numpy_array(graph)
        graph[graph!=0] = 1.0
        graph_nx = nx.from_numpy_array(graph)
        graph_sparse = scipy.sparse.csr_matrix(graph)
        n = int(graph.sum())
        not_graph_sparse = scipy.sparse.tril(graph_sparse).tocsr()
        mst = scipy.sparse.csgraph.minimum_spanning_tree(not_graph_sparse)
        mst[mst > 1] = 1
        mst.eliminate_zeros()
        ihatethis = not_graph_sparse - mst
        ihatethis = ihatethis.asformat("array")
        none = np.array([[0. for j in range(n)] for i in range(n)])
        return ihatethis == none

class DatasetLoader:
    def __init__(self, path: str):
        '''
        @params
        path: str --> the path to which the QM9 dataset will be downloaded
            note that the the code will complain if the path is incorrect,
            so make sure to double-check the path!
        '''
        self.dataset = QM9(path, transform=None, pre_transform=None, pre_filter=None)
        self.name = "Dataset Loader"

    def get_info(self, idx: int):
        '''
        gets the information of a specific molecule in a dataset
        @params
        idx: int --> the index of the element which you wish to view (0-indexed)
            raises an error if index is out of bounds
        '''
        data = self.dataset[idx]
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        return data
        
    def mol_info(self, data):
        '''
        similar to the get_info function
        '''
        for prop in data: print(prop)

    def draw_mol(self, data):
        '''
        draws the graphical representation of the molecule
        @params
        data: torch_geometric.data --> the molecule which you wish to be drawn
        '''
        vis = to_networkx(data)
        plt.figure(1, figsize=(8,8))
        nx.draw(vis)
        plt.show()

    def __len__(self) -> int:
        '''
        returns the number of molecules in the dataset
        '''
        return len(self.dataset) #13081

    def __getitem__(self, idx, verbose=False, draw=False):
        '''
        gets an item from the dataset
        @params
        idx: int --> the index of the item
            raises an error if the index is out of bounds
        verbose: bool (default False) --> if True, outputs the data associated with the item
        draw: bool (default False) --> if True, displays the graph of the molecule
        '''
        if verbose:
            self.mol_info(data)
            print()
        data = self.dataset[idx]
        self.get_info(idx)
        if draw:
            print()
            self.draw_mol(data)
        return data
    
class QM9Data(object):
    def __init__(self):
        '''
        Just your everyday wrapper class for getting QM9 data
        '''
        self.path = input("Enter the path where you want to download the QM9 dataset: ")
        self.DL = DatasetLoader(self.path)
        self.DP = DataProcessor()
        
    def __getitem__(self, idx, verbose=False, draw=False):
        '''
        gets an item from the dataset
        @params
        idx: int --> the index of the item
            raises an error if the index is out of bounds
        verbose: bool (default False) --> if True, outputs the data associated with the item
        draw: bool (default False) --> if True, displays the graph of the molecule
        '''
        try: assert(len(self.DL) > idx)
        except AssertionError: raise IndexError("Out of bounds!")
        sample = self.DL.__getitem__(idx, verbose=verbose, draw=draw)
        graph, matrix = self.DP.pyg_to_ssm(sample)
        if self.DP.molecule_is_own_MST(graph):
            print(f"The molecule at index {idx} is its own MST.")
            not_decided = True
            while not_decided:
                decide = input("Do you want to continue using this item or selected another item? Y/N: ")
                if decide == "Y" or decide == "N":
                    not_decided = False
                    if decide == "Y": return graph, matrix
                    else: self.__getitem__(int(input("Enter a new index: ")), verbose=verbose, draw=draw)
                else:
                    print("That's not an acceptable answer. Please try again!\n")
        return graph, matrix
                    
                