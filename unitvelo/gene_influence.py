#%%
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

def vals_to_csr(vals, rows, cols, shape):
    graph = coo_matrix((vals, (rows, cols)), shape=shape)
    graph_neg = graph.copy()

    graph.data = np.clip(graph.data, 0, 1)
    graph_neg.data = np.clip(graph_neg.data, -1, 0)

    graph.eliminate_zeros()
    graph_neg.eliminate_zeros()

    return graph.tocsr(), graph_neg.tocsr()

def get_iterative_indices(indices, index, n_recurse_neighbors=2):
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    return indices

def cosine_correlation(dX, Vi):
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = vector_norm(Vi)

    if Vi_norm == 0:
        result = np.zeros(dx.shape[0])
    else:
        result = np.einsum("ij, j", dx, Vi) / (norm(dx) * Vi_norm)[None, :]
        
    return result

def vector_norm(x):
    """computes the L2-norm along axis 1, equivalent to np.linalg.norm(A, axis=1)
    """
    return np.sqrt(np.einsum("i, i -> ", x, x))

def norm(A):
    """computes the L2-norm along axis 1, equivalent to np.linalg.norm(A, axis=1)
    """
    return np.sqrt(np.einsum("ij, ij -> i", A, A) if A.ndim > 1 else np.sum(A * A))

class Influence():
    def __init__(self) -> None:
        pass

    def get_n_jobs(self, n_jobs):
        import os

        if n_jobs is None or (n_jobs < 0 and os.cpu_count() + 1 + n_jobs <= 0):
            return 1, os.cpu_count()
        elif n_jobs > os.cpu_count():
            return os.cpu_count(), os.cpu_count()
        elif n_jobs < 0:
            return os.cpu_count() + 1 + n_jobs, os.cpu_count()
        else:
            return n_jobs, os.cpu_count()

    def get_indices(self, D):
        D.data += 1e-6
        A = D > 0
        n_counts = A.sum(1).A1 # summation over axis 1, equivalent to np.sum(A, 1)

        D.eliminate_zeros()
        D.data -= 1e-6
        indices = D.indices.reshape((-1, n_counts.min()))
        
        return indices, D

    def compute_cosines(self):
        vals, rows, cols = [], [], []

        for i in range(self.X.shape[0]):
            if self.V[i].max() != 0 or self.V[i].min() != 0:
                neighs_idx = get_iterative_indices(self.indices, i)

                dX = self.X[neighs_idx] - self.X[i, None]  # 60% of runtime
                dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[i])  # 40% of runtime

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * i)
                cols.extend(neighs_idx)

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0
        
        graph, graph_neg = \
            vals_to_csr(vals, rows, cols, shape=(self.X.shape[0], self.X.shape[0]))
        return graph, graph_neg

    def transition_matrix(self, gene_list):
        #! vgraph = VelocityGraph(adata, sqrt_transform=True, gene_subset=gene_list)
        used_genes = self.var_names.isin(gene_list)
        self.X = np.array(self.Ms[:, used_genes], dtype=np.float32)
        self.V = np.array(self.velocity[:, used_genes], dtype=np.float32)

        self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.indices = self.get_indices(self.distance)[0]

        #! vgraph.compute_cosines()
        graph, graph_neg = self.compute_cosines()

        #! transition matrix in sparse format
        graph = csr_matrix(graph)

        confidence = graph.max(1).A.flatten()
        ub = np.percentile(confidence, 98)
        self_prob = np.clip(ub - confidence, 0, 1)
        graph.setdiag(self_prob)

        T = np.expm1(graph * 10)  # np.exp(graph.A * 10) - 1
        T -= np.expm1(-graph_neg * 10)

        T = T.multiply(csr_matrix(1.0 / np.abs(T).sum(1))) # original `normalize` function
        T.setdiag(0)
        T.eliminate_zeros()
        return T.A

    def get_importance_simplify(self, args):
        gene_list, gene = args
        T = self.transition_matrix(gene_list)

        #! self.get_importance by clusters or entire dataset
        cosine = []
        cosine.append(np.abs(self.Tref - T).sum())

        for type in self.ctypes:
            index = np.squeeze(np.argwhere(self.label_list == type))
            cosine.append(np.abs(self.Tref[index, :] - T[index, :]).sum())
        
        return [cosine, gene]

    def verify_neighbors(self, adata):
        valid = "neighbors" in adata.uns.keys() and "params" in adata.uns["neighbors"]
        
        if valid:
            n_neighs = (adata.obsp["distances"] > 0).sum(1)
            valid = n_neighs.min() * 2 > n_neighs.max()

        if not valid:
            raise ValueError("You need to run scv.pp.neighbors first.")

    def recover_importance(
        self,
        adata,
        n_jobs=None
    ):
        """Rank genes importance (influence) on estimated velocity fields
            by filtering out one gene by another and compare the cosine similarity
            between original velocity embedding (equivalent to transition matrix)
            with the new velocity embedding minus one genes
        
        Args:
            adata (Anndata): 
                Anndata object.
            n_jobs (int): 
                number of CPU cores to use.
            
        Returns:
            dataframe: 
                calculated scores for each gene (unranked version)
                need to run `self.rank_importance(df)`
        
        """

        import time
        import multiprocessing
        self.verify_neighbors(adata)

        basis = adata.uns['basis']

        if f"X_{basis}" not in adata.obsm_keys():
            raise ValueError("You need to compute the embedding first.")

        if f'velocity_graph' not in adata.uns.keys():
            raise ValueError('Need to run `scv.tl.velocity_graph(adata, gene_subset=None)` first')

        vgenes = adata.var.loc[adata.var['velocity_genes'] == True].index
        n_jobs, total_jobs = self.get_n_jobs(n_jobs=n_jobs)

        self.adata = adata
        self.var_names = adata.var_names
        self.Ms = adata.layers["Ms"]
        self.velocity = adata.layers["velocity"]
        self.distance = adata.obsp["distances"]
        self.label_list = adata.obs[adata.uns['label']].values
        self.ctypes = list(set(adata.obs[adata.uns['label']].values))
        self.Tref = self.transition_matrix(list(vgenes))

        ctime = time.time()
        print (f"(using {n_jobs}/{total_jobs} cores)")
        with multiprocessing.Pool(n_jobs) as pool:
            self.res = pool.map_async(
                self.get_importance_simplify, 
                self.bufferize(vgenes)).get()
        print ('Time elapsed ', int(time.time() - ctime), ' seconds.')

        columns = ['Overall']
        columns.extend([str(type) for type in self.ctypes])
        df_aggre = pd.DataFrame(index=vgenes, data=0, dtype=np.float32, columns=columns)

        for _res in self.res:
            df_aggre.at[_res[1], :] = _res[0] 

        return (df_aggre - df_aggre.min(axis=0)) / \
            (df_aggre.max(axis=0) - df_aggre.min(axis=0))

    def bufferize(self, velocity_genes):
        buffer = []

        for gene in velocity_genes:
            used_genes = list(velocity_genes).copy()
            used_genes.remove(gene)

            buffer.append([used_genes, gene])
        
        return buffer

    def rank_importance(self, df):
        ranking = pd.DataFrame(index=list(range(len(df))), data=np.nan, 
            dtype=np.float32, columns=df.columns)
        
        for col in ranking.columns:
            ranking.at[:, col] = df.sort_values(by=[col], ascending=False).index
        
        return ranking
    
def influence(adata, n_jobs):
    import os
    from .gene_influence import Influence
    val = Influence()

    gene_score = val.recover_importance(adata, n_jobs)
    ranking = val.rank_importance(gene_score)

    gene_score = gene_score.sort_values(by=['Overall'], ascending=False)
    adata.uns['gene_rank'] = ranking
    adata.uns['gene_score'] = gene_score

    adata.write(os.path.join(adata.uns['temp'], 'temp.h5ad'), compression='gzip')

    return ranking, gene_score