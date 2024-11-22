# Lopit analyst
A program for analysing TMT labeled proteomics data that have been pre-processed with Proteome Discoverer 3.1+ (Thermo Scientific)

# Installation
CUDA/GPU requirements:
- Cuda 11+
- NVIDIA driver 450.80.02+
- Volta architecture or better (compute capability >=7.0)
- Python 3.11+

### 1. Create an environment with mamba (only valid in Waller lab server) and activate it:
```
source $HOME/.conda_source
replace <user> for your user account handle in the following command line:
mamba create --prefix=/wallerlab/opt/conda_dirs/<user>/envs/lopit_analyst python=3.11.0
conda activate lopit_analyst
```

### 2. Install CUDA standard (includes duDF, cuML, cuGraph, nx-cugraph, cuSpatioal, cuProj, cuxfilter, cuCIM, RAFT, cuVS)
```
mamba install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11.0 cuda-version=11.8
mamba install cuda-cudart cuda-version=11
```
**Note**: mamba is used to avoid known incompatibility issues between CUDA installation and conda

### 3. Install python modules using pip.  
pip install <module>
modules required:
- PyPDF2
- dash
- dask
- hdbscan
- matplotlib
- missingno
- openpyxl
- pypdf
- seaborn
- patchworklib==0.6.2

### 4. LA modules: (make it easy-> distributee as conda env)
-


#Usage

## 1 Input files
Required files:
Outputs from PD 3.1
-	**Psm file**: PD file suffixed with ‘PSMs.txt’
-	**Protein file**: PD file suffixed with ‘Proteins.txt’
Prepared by user:
-	**Phenotypic file**: It is a file containing information regarding the experiment and it must contain the following columns: ‘Sample.name’, ‘Experiment.name’, ‘Tag’, ‘Tag.name’, ‘Gradient.type’, ‘Peptide.amount’.  Peptide.amount column indicates the amount of total peptide for each channel before pooling the sample for mass spectrometry (e.g., PL_pData.tsv). 
#### Optional files:
-	**Markers** (optional for step 5 but mandarory for step 6): tsv file containing marker accessions to be used during supervised machine learning. However, this file must contain ALL accessions that appear in the predicted proteome files used for protein identification in PD. This is because any detected protein found in the proteomics experiment is reported by PD in the psm file.  These accessions should correspond to predicted nuclear, mitochondrial, plastid and/or symbiont proteins (fasta files) either single or either combination.  Format: two columns: ‘Accession’, ‘marker’   Note: any accession that is not labelled as marker must be labeled as ‘unknown’ (case sensitive)
-	**Proteins Feature** : tsv file containing the paths to outputs of Signal Peptide (e.g., Pmarinus.hq.2022_02_02.pred_res_2023_06_20.tsv), Target Peptide (e.g., Pmarinus.hq.2022_02_02_targetp2.tsv), deepTMHMM (e.g., Pmarinus.deeptmhmm.tmhmm.first-tm.tsv), Deeploc (e.g., Perkinsus_marinus.deeploc.140424.tsv), phobius (e.g., Pmarinus.hq.2022_02_02.Phobius.short.out). Format: two tab separated columns: ‘Type’, ‘Path’. See file called Sequence_properties.tsv
    #### Note: 
    - Outputs must contain each accession’s information in a single line  (aka short outputs)
    - deepTMHM file must be preprocessed and contain two columns (e.g., Pmarinus.deeptmhmm.tmhmm.first-tm.tsv)
    - the program will accept one, two or several of the Signal Peptide, Target Peptide, deepTMHMM, Deeploc, phobius inputs. These inputs must be in ‘short’ format (all the information for one accession must be in a single line)
-	**Annotation**: file containing all known annotations for each accession. Please make sure headers do no contain spaces, dots (‘.’) or any special character (e.g., Pmarinus.eval0.001.PTHR17-Score.tsv)
-	**Tagm**: tsv file containing TAGM classification (e.g., classification for 4-way experiment combination PLNPLOPL2PL1.TAGM-MAP.out.tsv)

## 2 Environment activation
```
source $HOME/.conda_source
export PATH=$PATH:/home/<user>/Lopit_Analyst_Source  #  replace user by your own username
conda activate lopit_analyst
```
## 3. Workflow description:
  - Preparation of input files (data_prep), 
  - Step 1. Diagnostics (diagnostics)
  - Step 2. Removal of contaminants, low quality psms, etc (filtering)
  - Step 3. Removal of missing values (mv_removal)
  - Step 4. Missing value imputation and psm aggregation (imputation_aggregation)
  - Step 5. Dimensionality reduction and clustering (clustering)
  - Step 6. Supervised machine learning classification (sml)

# Study case:
To illustrate how to use the program,  the Perkinsus marinus dataset is used and is available at /wallerlab/storage/dsalas/PD3.1_14082024_example.  
### Experiment description:
#### PL1 and PL2.  
Trophozoites were lysed by nitrogen cavitation (hypotonic conditions: ~300 mOsm). The lysed material was separated in membrane and soluble fractions; the PL1 membrane fraction was separated with a self-established density gradient of iodixanol (16 % w:v) and the PL2 one was separated with a pre-formed density gradient. Fractions of each experiment were collected, pooled and each experiment was labeled with a TMT10plex.
#### PLO. 
Trophozoites were lysed by nitrogen cavitation (isotonic conditions ~750 mOsm). The lysed material was separated with a pre-formed density gradient. Odd and even fractions of the resolving gradient were pooled into two 10plexes called PLO1 and PLO2, using 10 of the 11 channels of a TMT11plex; the 11 channel was reserved for the bridging sample containing approximately equal proportion of peptides taken from each fraction across the gradient.
#### PLN. 
Same experimental design as PLO, but the homogenisation and gradient media were supplemented with K+ and Mg2+ salts.
Note that given the PLO and PLN experimental design they need to be bioinformatically reconstituted using ‘Internal Reference Scaling’ into single experiments each (PLO and PLN.
### Proteome discoverer (3.1) processing of raw mass spectra:
The raw mass spectra were searched against 17278 non-redundant predicted protein sequences from a newly obtained genome of P. marinus ATCC 50983. Culture was axenic, hence additional taxa contaminant protein files were provided (other than cRAP). Peptide-to-spectrum match was performed using Mascot with FDR validation via Percolator. Only PSMs at FDR ≤ 1 % were retained. Peptide grouping, protein inference, and protein grouping were carried out under strict parsimony. TMT quantification was done with 20 ppm mass tolerance and the most confident centroid for the reporter ion peak detection. TMT abundances were reported as signal-to-noise ratios. No filters were applied. 



# Preparation of input files 
Recipes for preparing Perkinsus marinus input files
- Source files are available at /wallerlab/storage/dsalas/PD3.1_14082024_example
## Preparing PSMs file:
```
lopit_analyst.py data_prep --data_type psms \
                           --input PL12NO_PSMs.txt \
                           --out_name Pmar \
                           --rename_columns 'PL1:TMT131-TMT131N;PL2:TMT131-TMT131N' \
                           --use_column 'File.ID' \
                           --experiment_prefixes 'PL1-F1;PL2-F2;PLN1-F3;PLN2-F4;PLO1-F5;PLO2-F6'
```
### Outputs produced:
Within newly created ‘Formatted_input_data_Pmar’ directory:
- If '--rename_columns' passed:  Pmar_formatted_PSMs.matrix_w_re-declared_channels.tsv
- If '--rename_columns' not passed: Pmar_unformatted_PSMs.matrix.tsv
#### Note:
1)	To understand the format needed in ‘rename_columns’ and ‘experiment_prefixes’ type:
lopit_analyst.py data_prep -h
2)	The following recipes will format files and deposit them in a directory prefixed by ‘Formatted_input_data’ and suffixed with the output string you specify (in example:  ‘Pmar’)
3)	The formatted files will be prefixed with the output string you specify (in example:  ‘Pmar’)

## Preparing phenotypic file:
```
lopit_analyst.py data_prep --data_type pheno_data \
                           --input PL_pData.tsv \
                           --out_name Pmar
```
### Outputs produced: 
Within newly created ‘Formatted_input_data_Pmar’ directory:
- Pmar_formatted_phenodata.tsv

## Format of protein information:
```
lopit_analyst.py data_prep --data_type protein_data \
                           --input PL12NO_Proteins.txt \
                           --out_name Pmar \
                           --search_engine Mascot
```
### Outputs produced: 
Within newly created ‘Formatted_input_data_Pmar’ directory:
- Pmar_formatted_protein_data.tsv

## Format of sequence features:
```
lopit_analyst.py data_prep --data_type protein_features \
                           --input Sequence_properties.tsv  \
                           --out_name Pmar
```
### Outputs produced: 
Within newly created ‘Formatted_input_data_Pmar’ directory:
- Pmar_PD31_14082024_paper2_formatted_protein_features.tsv

-------------

### Important: 
The Following command lines assume you are in the same directory where the Formatted_input_data_Pmar directory was created. If not, you must provide absolute paths to the files

# Step 1 Diagnostics
This step will provide several diagnostic plots that can be used to make decisions about downstream data processing.
```
lopit_analyst.py diagnostics --input Formatted_input_data_Pmar/Pmar_formatted_PSMs.matrix_w_re-declared_channels.tsv \
                             --additional_file Formatted_input_data_Pmar/Pmar_formatted_phenodata.tsv \
                             --out_name Pmar
```
### Outputs produced:
Within newly created ‘Step1__Diagnostics_ Pmar’ directory:
#### Plots:
##### All_diagnostic_hist.pdf:
-	Distribution of precursor mass errors and mass accuracy (PSMs delta masses)
-	Distribution of precursor ion intensities (Intensity)
-	Percentage of isolation interference (Isolation interference)
-	Percentage of SPS fragments that were matched to peptide fragments for a given PSM (SPS mass matches)
-	Distribution of the average signal-to-noise ratio of reporter ion peaks (Average reporter SN ratio)
-	Time used by the mass spectrometer to accumulate precursor ions in an ion trap (Ion injection time)
-	Variability in the precursor mass accuracy across the run time (Retention time vs delta M)
-	Distribution of the TMT reporter ion intensities (PSMs)
-	Distribution of raw TMT abundances (notch artifact)
##### P.All.Exp_PeptideAbund.trend.pdf:
-	Sum of TMT abundances by channel plotted against the peptide amount labelled.
##### TMT.Abundance.by.Exp.pdf:
-	TMT abundances by experiment
##### Labeling efficiency
Excel files with labeling efficiency calculations by experiment.
#### Main output:
Parsed_PSM.headers.Pmar.tsv 

# Step2 data basic filtering of low-quality PSMs
This step will **retain** unambiguous PSMs, PSMs that match a unique protein group, PSMs of rank = 1, PSMs of rank = 1 by search engine, target and decoy PSMs matched to a given spectrum and ranked by their score PSMs identified for MS2-spectra, and PSMs with injection time < 50 ms. Simultaneously, it will **remove** PSMs annotated as contaminants and provided to the ‘--exclude_taxon’ flag, PSMs with isolation interference > 50%, PSMs with fewer than half SPS precursors matched, PSMs with low S/N, entries == "NoQuanLabels"

```
lopit_analyst.py filtering --input Step1__Diagnostics_ Pmar/Parsed_PSM.headers.Pmar.tsv \
                           --out_name Pmar \
                           --exclude_taxon other
```
#### Note: 
The flag '--exclude_taxon' corresponds to a known contamination source in the file (host, symbiont, etc). Such taxon(taxa) must have been specified before running PD [If there were multiple contaminants, they must have been specified with a single handle (e.g., other_taxa)]. In the example, the argument ‘other’ is passed as the data comes from an axenic culture and no additional taxa are expected in the PSMs file to bypass this requirement (if unsure, check PSMs column called ‘Marked.as’). **WARNING:** not specifying this flag correctly will lead to an inaccurate filtering.

### Outputs produced: 
Within newly created ‘Step2__ First_filter_Pmar’ directory:
#### Plots:
##### Comparative_plots.filter1.pdf:
-	Tag abundance by experiment pre and post filtering
-	TMT avundance by experiment pre and post filtering
### Main output:
- Filtered data file: mv_calculated_full_df.tsv
- Master protein accessions report pre and post filtering by experiment: Filtering_1-report.tsv


# Step 3. Removal of missing values (mv_removal)
This step allows the exploration of the patterns of missing values by experiment and allows to set a threshold to remove TMT channels with missing values. Note that none of the machine learning methods used in this workflow can handle missing values (MVs). Hence, MVs should be either removed and/or imputed using an adequate strategy. This means that when a condition is met some columns or rows of the data are deleted when they contain missing values and others will be imputed (e.g., mean, minimum, zero or model-based methods). This step will only remove PSMs with missing values in cases when there are PSMs with complete data for the same peptide sequence and remove TMT channels that are above a user specified threshold.
```
lopit_analyst.py mv_removal --input Step2__First_filter_Pmar/mv_calculated_full_df.tsv \
                            --out_name Pmar \
                            --remove_columns 0.1
```
#### Note: 
Remove-columns will eliminate TMT channels that contain the specified proportion (in this case 0.1 =  10%). This flag is optional, and it could be as astringent as desired (values between 0 - 1)

### Outputs produced: 
Within newly created ‘Step3__Missing_data_figures_Pmar’: pre and post PSMs removal plots
- Comparative_heatmaps_pre.pdf and Comparative_heatmaps_post.pdf
- Comparative_mv_cluster_tree_pre.pdf and Comparative_mv_cluster_tree_post.pdf
- Comparative_mv_correlation_heatmap_pre.pdf and Comparative_mv_correlation_heatmap_post.pdf
- Comparative_mvals_boxplots_pre.pdf and Comparative_mvals_boxplots_post.pdf
- Comparative_pie_charts_of_missing_values-pre.pdf
- Total_MV_by_TMT_channel_and_total_protein_groups

Within newly created ‘Step3__DF_ready_for_mv_imputation_Pmar’:
- Filtered_df-ready-for-imputation.protein.tsv
- Filtered_df-ready-for-imputation.peptide.tsv

# Step 4. Missing value imputation and psm aggregation (imputation_aggregation)
Data can be missing completely at random (MCAR), missing at random (MAR), and/or  missing not at random (MNAR). To determine how MVs occur is necessary to know how the data were generated. This workflow uses MinDet (deterministic minimal value using quantiles a minimal values) for MNAR and KNN (K-Nearest Neighborg) for MAR.
#### How to deal with MVs:
### MNAR:
Some organelles may be present in a specific density range(s), which means that fractions outside that range may completely lack of such an organelle proteins. For example, the highest density fraction may be devoid of organelles that will be in the lowest density fractions or the soluble fraction, and the soluble fraction (cytosol) in a HyperLopit experiment may exclude organelles from the insoluble fractions.
### MAR: 
Generally, missing values in proteomics are due to the stochastic nature of data-dependent acquisition methods and imperfect detection efficiency of mass spectrometers, and when TMT labeling is involved, then quantification data, poor labelling efficiency in some channels, as well as insufficient intensity and poor ion statistic may lead to MVs. Note that the MVs  probability depends on peptide abundance.

- In the Perkinsus data, channels TMT126 (high density fraction) and TMT131N (cytosolic fraction) are expected to be MNAR. The remaining channels are expected to be MAR.  Note that after MV imputation, PSMs will be aggregated accordingly to their master protein accessions and experiment, 
 and given the experimental design we need to reconstitute the experiments PLO1 and PLO2, and PLN1 and PLN2 into single PLO and PLN experiments.  Aggregation uses PSMs median values.


\#missing value imputation and aggregation of PSMs by master protein accession:
```
lopit_analyst.py imputation_aggregation --input Step3__DF_ready_for_mv_imputation_Pmar/Filtered_df-ready-for-imputation.protein.tsv  \
                                        --out_name Pmar_protein_level \
                                        --accessory_data Formatted_input_data_Pmar/Pmar_formatted_phenodata.tsv \
                                        --protein_data Formatted_input_data_Pmar/Pmar_formatted_protein_data.tsv \
                                        --mnar MinDet \
                                        --channels_mnar 'PL1-PL2-PLN1-PLN2-PLO1-PLO2:TMT126,TMT131N' \
                                        --mar knn \
                                        --channels_mar 'PL1-PL2-PLN1-PLN2-PLO1-PLO2:remainder' \
                                        --interlaced_reconstitution 'PLN:PLN1,PLN2;PLO:PLO1,PLO2'
```
### Outputs produced: 
Within newly created ‘Step4__PSM_Normalization_Pmar_acc’:
#### Plots:
- Comparative_boxplots.jpg
- Comparative_boxplots-with_reconstitution.jpg
- VennDiagramPLN2-PLN1-PLO2-PL2-PLO1-PL1.pdf
#### Main output: 
Missing value imputed PSMs and PSMs aggregated by accession separated by experiment: 
- df_for_clustering_PL1.tsv, df_for_clustering_PL2.tsv, df_for_clustering_PLN.tsv, df_for_clustering_PLO.tsv


\#missing value imputation and aggregation of PSMs by accession and PMS sequence (peptide level):
```
lopit_analyst.py imputation_aggregation --input Step3__DF_ready_for_mv_imputation_Pmar/ Filtered_df-ready-for-imputation.peptide.tsv \
                                        --out_name Pmar_peptide_level \
                                        --accessory_data Formatted_input_data_Pmar/Pmar_formatted_phenodata.tsv \
                                        --protein_data Formatted_input_data_Pmar/Pmar_formatted_protein_data.tsv \
                                        --mnar MinDet \
                                        --channels_mnar 'PL1-PL2-PLN1-PLN2-PLO1-PLO2:TMT126,TMT131N' \
                                        --mar knn \
                                        --channels_mar 'PL1-PL2-PLN1-PLN2-PLO1-PLO2:remainder' \
                                        --interlaced_reconstitution 'PLN:PLN1,PLN2;PLO:PLO1,PLO2'
```
### Outputs produced: 
Within newly created ‘Step4__PSM_Normalization_Pmar_acc’:
#### Plots:
- Comparative_boxplots.jpg
- Comparative_boxplots-with_reconstitution.jpg
- VennDiagramPLN2-PLN1-PLO2-PL2-PLO1-PL1.pdf
#### Main output: 
Missing value imputed PSMs and PSMs aggregated by accession-psms separated by experiment: 
- df_for_clustering_PL1.tsv, df_for_clustering_PL2.tsv, df_for_clustering_PLN.tsv, df_for_clustering_PLO.tsv
#### Notes:
1)	Format needed in ‘--channels_mnar’, ‘--channels_mar’, and ‘--interlaced_reconstitution’ flags is explained in help menu (lopit_analyst.py imputation_aggregation -h)
2)	Interlaced reconstitution only applies to experiments such as the one described for Perkinsus marinus
3)	missing value imputation and aggregation of PSMs by accession and PMS sequence will take 2 or 3 times more time than missing value imputation and aggregation of PSMs by master protein accession to be completed. 


# Step 5. Dimensionality reduction and clustering (clustering)
This step will carry out dimensionality reduction via tSNE and UMAP (both done on the TMT expression data), as well as unsupervised clustering using HDBSCAN (directly on the TMT expression data and on the UMAP embedding coordinates). There are several flags that are specific to each method that need to be declared in the command line. For more information type: lopit_analyst.py clustering -h

\# Missing value imputation and aggregation of PSMs by master protein accession:
```
lopit_analyst.py clustering --input Step4__PSM_Normalization_Pmar_acc/df_for_clustering \
                            --out_name Pmar_protein_level \
                            --group_combinations all \
                            --method_tsne exact \
                            --perplexity 50 \
                            --cluster_selection_epsilon 0.025 \
                            --min_size 6 \
                            --min_dist 0.25 \
                            --markers_file Pmar_385markers.19359.12112024.capitalized.tsv \
                            --protein_features Formatted_input_data_Pmar/Pmar_formatted_protein_features.tsv
```
### Outputs produced: 
Within newly created ‘Step5__Clustering_Pmar_acc’:
  New directories per each experiment combination and within each directory:
- tSNE directory: <Experiment combination>_t-SNE.plot_2c_50.pdf
- UMAP directory: <Experiment combination>_2-dims_UMAP_dim1_2cUMAP_dim2_2c.plot.pdf
- HDBSCAN directory: persistence and stats by TMT expression data and UMAP. Experiment combination classification using Euclidean and Manhattan distances by TMT expression data and UMAP. Sumaries TMT expression data and UMAP.
- Coordinates_ALL_<experiment_combination>_df.tsv: data containing tSNE, UMAP, HDBSCAN results
- If protein features were provided the main output is Final_df_<experiment_combination>.tsv data containing tSNE, UMAP, HDBSCAN results and appended protein features. Otherwise, Coordinates_ALL_<experiment_combination>_df.tsv is the file to be used in the next step.

\# Missing value imputation and aggregation of PSMs by master protein accession and psm sequence (peptide level):
```
lopit_analyst.py clustering --input Step4__PSM_Normalization_Pmar_peptide_level/df_for_clustering \
                            --out_name Pmar_peptide_level \
                            --group_combinations all \
                            --method_tsne exact \
                            --perplexity 50 \
                            --cluster_selection_epsilon 0.025 \
                            --min_size 6 \
                            --min_dist 0.25 \
                            --markers_file Pmar_385markers.19359.12112024.capitalized.tsv \
                            --protein_features Formatted_input_data_Pmar/Pmar_formatted_protein_features.tsv
```
### Outputs produced: 
Within newly created ‘Step5__Clustering_Pmar_peptide_level’:same outputs as above but by peptide.

#### Notes:
1)	UMAP coordinates can be used as input for clustering and machine learning because UMAP is a deterministic method (unlike t-SNE)
2)	PCA is not automatically calculated. To do so, you must enable the flag --pca  True. Multidimensional UMAP is time consuming, and it needs to be enabled in the command line.
3)	Input will take the shared prefix of the files obtained in the previous step. In this case is ‘df_for_clustering’
4)	If --feature_projection is enabled: projections are generated for tSNE, and UMAP. These will contain HDBSCAN clusters and markers (if markers provided). Also, if a valid --protein_features file is provided additional projections by features (Signal peptide, Target peptide, etc) will be generated.
5)	if you are using an input that was not obtained with the current wokflow, you input must contain the following columns: 'Accession', 'calc.pI', 'Dataset', 'Number.of.PSMs.per.Protein.Group', and all expression data starting with the word TMT (e.,g.,'TMT126', 'TMT134N_1', 'TMT134N_2'... etc) 


# Step 6. Supervised machine learning classification (sml)
Four supervised machine learning methods are implemented: Support vector machine (SVM), K-Nearest Neighbors (KNN), Random Forest (RF), and Naïve Bayes (NB). For each method the best hyperparameters for a given dataset are obtanined and used for training and classification. All of the methods are estimated for all experiment combinations obtained in Step5. The markers provided for training can be used as they are specifying the method ‘unbalanced’ to the ‘--balancing_method’ flag. However, the variable number of proteins working in different organelles, hence, the variable number of available markers per organelle can create biases during classification process. A way to mitigate such a bias is to use a balanced training set via generation of synthetic markers (e.g., ‘borderline’, ‘over_under’ or ‘smote’).  a Three and four majority rule classification is also generated using HDBSCAN classification and the four methods.  TAGM classification may be added at a later stage if provided, then HDBSCAN is replaced for TAGM for the three and four majority rule classification.

\# SML by master protein accession:
```
lopit_analyst.py sml --input Step5__Clustering_Pmar_protein_level \
                     --out_name Pmar_protein_level_smote \
                     --recognition_motif  Final_df_P* \
                     --markers_file  Pmar_385markers.19359.12112024.capitalized.tsv \
                     --balancing_method smote \
                     --accessory_file Pmar_385markers.19359.12112024.capitalized.tsv
```
### Outputs produced: 
Within newly created ‘Step6__SML_predictions_Pmar_protein_level’:
A new directory for each directory in Step5 is created containing:
### Maing output:
- Final_df_<experiment_combination>.SML.Supervised.ML.tsv
- Precision matrices and classification reports (precision, recall, f1-score, support):
    - SVM.accuracy.estimations.xlsx	
    - KNN.accuracy.estimations.xlsx
    - Random_forest.accuracy.estimations.xlsx
    - Naive_Bayes.accuracy.estimations.xlsx


\# SML by master protein accession and psms sequence:
```
lopit_analyst.py sml --input Step5__Clustering_Pmar_accpsm \
                     --out_name Pmar_peptide_level_smote \
                     --recognition_motif  Final_df_P* \
                     --markers_file  Pmar_385markers.19359.12112024.capitalized.tsv \
                     --balancing_method smote \
                     --accessory_file Pmar_385markers.19359.12112024.capitalized.tsv
```
### Outputs produced: 
Within newly created ‘Step6__SML_predictions_Pmar_accpsm’: same outputs as above but by accession-psm
### Note:
1)	if you are using an input that was not obtained with the current wokflow, you input must contain the following columns: 'Accession', and all the expression data starting with the word TMT (e.,g.,'TMT126', 'TMT134N_1', 'TMT134N_2'... etc).  The input file must be inside a directory within a main directory (emulating the directory architecture that is generated by Step5 of this workflow)
