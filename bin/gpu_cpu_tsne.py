import sys
import os
import pandas as pd
import graphic_results as gr
from typing import Optional
import gpu_memory_management as gmm



def gpu_tsne_safe(df: pd.DataFrame, dataset: str,
                  perplexity: float = 30.0,
                  n_iter: int = 1000, learning_rate: float = 200.0,
                  method: str = 'barnes_hut', components: int = 2) -> Optional[
    pd.DataFrame]:
    """GPU t-SNE with memory management"""
    if not gmm._gpu_available:
        return None

    try:
        import cudf
        from cuml.manifold import TSNE

        print(f't-SNE via GPU for dataset: {dataset}')
        cols = df.columns.to_list()
        odf = df.copy(deep=True).reset_index()
        ndf = odf.loc[:, cols].copy(deep=True)
        cdf = cudf.DataFrame.from_pandas(ndf)



        if method == 'barnes_hut':
            theta = 0.5
            tsne = TSNE(n_components=components, random_state=1155,
                        learning_rate=300, init='random', perplexity=perplexity,
                        method=method, n_iter=5000, angle=theta,
                        exaggeration_iter=300,
                        n_neighbors=3 * perplexity)
        else:
            tsne = TSNE(n_components=components, random_state=1155,
                        learning_rate=300.0, init='random',
                        perplexity=perplexity,
                        method=method, n_iter=10000, exaggeration_iter=300,
                        n_neighbors=3 * perplexity)

        tsne_result = tsne.fit_transform(cdf)
        embedding_df = tsne_result.to_pandas()
        #  -- creating a dataframe with the 2 new dimensions -- #

        embedding_df.rename(columns={0: f'tSNE_dim_1_2c_{perplexity}',
                                     1: f'tSNE_dim_2_2c_{perplexity}'},
                            inplace=True)

        indexes = zip(odf['Dataset'], odf['Accession'],
                      odf['calc.pI'], odf['Number.of.PSMs.per.Protein.Group'])
        index_names = ['Dataset', 'Accession', 'calc.pI',
                       'Number.of.PSMs.per.Protein.Group']
        embedding_df.index = pd.MultiIndex.from_tuples(indexes,
                                                       names=index_names)

        # Clean up
        del cdf, tsne, tsne_result
        gmm.cleanup_gpu_memory()
        return embedding_df

    except Exception as e:
        print(f"GPU t-SNE failed for {dataset}: {e}")
        gmm.cleanup_gpu_memory()
        return None


