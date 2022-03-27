import numpy as np

class Confidence():
    def __init__(self, 
        base_iter=5,
        threshold=0.5,
    ):
        self.base_iter = base_iter
        self.threshold = threshold

    def get_gene_subset(self, temp):
        idx = temp.var['velocity_genes'].values.copy()
        ref = np.where(idx == True)[0]
        size = int(len(ref) * self.threshold)

        sampling = np.random.choice(ref, size=size, replace=False)
        idx[sampling] = False
        temp.var['velocity_genes'] = idx
        return temp

    def get_cell_confidence(self, temp):
        import itertools
        combinations = list(itertools.combinations(range(self.base_iter), 2))

        confident = 0
        for combo in combinations:
            vect_one = temp.obsm[f's_{combo[0]}']
            vect_two = temp.obsm[f's_{combo[1]}']

            numerator = np.sum(vect_one * vect_two, axis=1)
            denominator = np.linalg.norm(vect_one, axis=1) * \
                np.linalg.norm(vect_two, axis=1)
            
            cosine = numerator / denominator
            confident += (cosine / len(combinations))
        
        return (confident - np.min(confident)) \
            / (np.max(confident) - np.min(confident))

def confidence(adata, base_iter=5, threshold=0.5):
    """Preparation and pre-processing function of RNA velocity calculation.
    
    Args:
        adata (Anndata):
            processed adata object
        base_iter (int): 
            number of random sampling and base combinations of subset
        threshold (float): 
            percentage of genes filtered out for calculating new velocity graph
        
    Returns:
        None
    
    """

    if adata.uns['examine_genes']:
        raise ValueError(
            f'self.EXAMINE_GENE in configuration file should be False.\n'
            f'Please re-run the model use alternative setting.'
        )

    else:
        import scvelo as scv
        conf = Confidence(base_iter, threshold)
        conf_data = adata.copy()
        basis = adata.uns['basis']

        for iter in range(base_iter):
            subge = conf.get_gene_subset(conf_data.copy())
            scv.tl.velocity_graph(subge, sqrt_transform=True)
            scv.tl.velocity_embedding(subge, basis=basis)
            conf_data.obsm[f's_{iter}'] = subge.obsm[f'velocity_{basis}'][:, :2]

        col_confidence = conf.get_cell_confidence(conf_data)
        adata.obs['confidence'] = col_confidence

        scv.pl.scatter(adata, color='confidence', cmap='magma', 
            basis=basis, dpi=300)