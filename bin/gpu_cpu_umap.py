import os
import sys
import numpy as np
import pandas as pd
import graphic_results as gr
import gpu_memory_management as gmm
from typing import Optional



def gpu_umap_safe(dfs: tuple, dataset: str,
                  n_neighbors: int = 15, min_dist: float = 0.1,
                  n_components: int = 2) -> Optional[pd.DataFrame]:
    """GPU UMAP with memory management"""
    if not gmm._gpu_available:
        return None

    try:
        import cudf
        from cuml import UMAP
        from cuml.preprocessing import StandardScaler as cuStandardScaler

        print(f'UMAP via GPU for dataset: {dataset}')

        def create_umap(df, n_components,  min_dist, n_neighbors):
            ndf = df.copy(deep=True)
            original_index = ndf.index
            cdf = cudf.DataFrame.from_pandas(ndf)

            # Scale data
            scaler = cuStandardScaler()
            scaled_data = scaler.fit_transform(cdf)

            # Adjust n_neighbors if necessary
            if n_neighbors is None:
                n_neighbors = int(np.ceil(np.sqrt(df.shape[0])))

            # Ensure n_neighbors is not larger than dataset size
            if n_neighbors > ndf.shape[0]:
                n_neighbors = min(n_neighbors, ndf.shape[0] - 1)
                print(f'neighbors declared are larger than dataset size. '
                      f'Setting to {n_neighbors}')

            umap_gpu = UMAP(n_components=n_components, min_dist=min_dist,
                            n_neighbors=n_neighbors, init='random',
                            n_epochs=1000, random_state=1661, verbose=False)

            umap_result = umap_gpu.fit_transform(scaled_data)

            # resulting df containing UMAP coordinates converted to pandas
            result_df = umap_result.to_pandas()

            # Create appropriate column names
            if n_components == 2:
                result_df.columns = ['UMAP_dim1_2c', 'UMAP_dim2_2c']
            else:
                result_df.columns = [f'UMAP_dim{i + 1}_{n_components}c' for i in
                                     range(n_components)]
            result_df.index = original_index

            # Clean up
            del cdf, scaler, scaled_data, umap_gpu, umap_result
            gmm.cleanup_gpu_memory()
            return result_df


        def my_umap(dfs, min_dist, n_neighbors):
            df, pca_df = dfs
            # Generating embeddings for multiple components

            print(f'UMAP embeddings for 2 components')
            umap_2dim_df = create_umap(df=df,
                                       n_components=2,
                                       min_dist=min_dist,
                                       n_neighbors=n_neighbors)
            # set minimum distance very low (e.g., 5e-324) for density-based clustering
            if pca_df is None:
                print(f'UMAP embeddings for 3 components')
                # was set to pca
                umap_xdim_df = create_umap(df=df,
                                           n_components=3,
                                           min_dist=min_dist,
                                           n_neighbors=n_neighbors)
            else:
                print(f'UMAP embeddings for {pca_df.shape[1]} pca components')
                umap_xdim_df = create_umap(df=pca_df,
                                           n_components=pca_df.shape[1],
                                           min_dist=5e-324,
                                           n_neighbors=n_neighbors)

            umap_all_dims_df = pd.merge(umap_2dim_df, umap_xdim_df,
                                        left_index=True,
                                        right_index=True, how='inner')
            return umap_all_dims_df
        return my_umap(dfs, min_dist, n_neighbors)

    except Exception as e:
        print(f"GPU UMAP failed for {dataset}: {e}")
        gmm.cleanup_gpu_memory()
        return None
