import os
import pandas as pd
import graphic_results as gr
from typing import Optional
import gpu_memory_management as gmm
import lopit_utils



def gpu_pca_safe(df: pd.DataFrame, dataset: str, comp: float = 0.99) -> \
Optional[pd.DataFrame]:
    ''' GPU PCA with memory management and error handling '''
    if not gmm._gpu_available:
        return None

    ndf = df.copy(deep=True)

    try:
        import cudf
        from cuml.decomposition import TruncatedSVD

        print(f'PCA via GPU for dataset: {dataset}')

        # Convert to GPU DataFrame
        cdf = cudf.DataFrame.from_pandas(ndf)

        # Iterative approach to find optimal components
        explained_variance_ratio_ = 0
        n_components = 1
        desired_variance = comp

        while (explained_variance_ratio_ < desired_variance and
               n_components < min(cdf.shape)):
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(cdf)
            explained_variance_ratio_ = svd.explained_variance_ratio_.sum()
            n_components += 1

        n_components -= 1  # adjust for the last increment in the loop
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(cdf)
        transformed_data = svd.transform(cdf)

        # Convert back to pandas
        transformed_df = transformed_data.to_pandas()
        ncolnames = {col: f'PC{int(col) + 1}' for col in
                     transformed_df.columns.to_list()}
        transformed_df.rename(columns=ncolnames, inplace=True)
        transformed_df.index = df.index
        # transformed_df.to_csv(f'PCA_results_{dataset}.tsv',
        #                       sep='\t', index=True)
        # Clean up GPU objects
        del cdf, svd, transformed_data
        gmm.cleanup_gpu_memory()
        return transformed_df

    except Exception as e:
        print(f"GPU PCA failed for {dataset}: {e}")
        gmm.cleanup_gpu_memory()
        return None

