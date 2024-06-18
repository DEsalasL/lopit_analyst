import os
import re
import sys
import lopit_utils
import numpy as np
import pandas as pd
from functools import reduce




#   ---   headers beginning ---
headers = {
    'signalp': ['Accession', 'signalp-pred', 'SP(Sec-SPI)', 'signalp-other',
           'signalp-CS-Position'],
    'targetp': ['Accession', 'targetp-pred', 'targetp-noTP', 'targetp-SP',
           'targetp-mTP', 'targetp-CS-Position'],
    'phobius': ['Accession', 'phobius-TM', 'phobius-SP', 'phobius-prediction'],
    'tmhmm': ['Accession', 'length', 'tmhmm-Exp', 'tmhmm-First60',
         'tmhmm-PredHel', 'tmhmm-Topology'],
    'deeptmhmm': ['check source headers'],
    'gpianchored': ['Accession', 'length', 'GPI-Anchored-pred',
                    'gpi-Omega-site pos.', 'gpi-Likelihood', 'gpi-Amino-acid']}


#  ---   headers end   ---
def processing_list(list_in):
    df = pd.read_csv(list_in, header=0, sep='\t')
    dic = dict(zip(df.Type, df.Path))
    return dic


def flat_value(txt_list):
    possitions = []
    for i in txt_list:
        if '-' in i:
            a, b = i.split('-')
            possitions.append((int(a), int(b)))
        else:
            possitions.append(i)
    return possitions


def get_phobius_values(df, coln):
    original_cols = df.columns.to_list()
    col_names = ['phobius-sp', 'CS position-phobius', 'first-tm-tmp']
    df[col_names] = pd.DataFrame(df.tmp.to_list(), index=df.index)
    df[coln] = df['first-tm-tmp'].apply(lambda x:
                                        x if isinstance(x, tuple) and
                                        x[1] <= 80 else '')
    del df['first-tm-tmp']
    # -- sliced df by cols  --  #
    sel_cols = original_cols + col_names[:2] + [coln]
    ndf = df.loc[:, sel_cols].copy(deep=True)
    return ndf


def apply_filter(df, typ):
    coln = 'phobius-first-tm <= 80aa'
    if typ == 'sp':
        df['tmp'] = df['phobius-Prediction'].apply(
            lambda x: flat_value(re.findall(r'\d+\-\d+|\d+\/\d+', x)[:3]))
        return get_phobius_values(df, coln)
    else:
        df['first-tm-tmp'] = df['phobius-Prediction'].apply(
            lambda x: flat_value(re.findall(r'\d+\-\d+|\d+\/\d+', x))[0])
        df[coln] = df['first-tm-tmp'].apply(lambda x:
                                            x if isinstance(x, tuple) and
                                            x[1] <= 80 else '')
        del df['first-tm-tmp']
        return df


def parse_phobius(df):
    #  -- df containing only sp + tp predictions --  #
    nopred_cond = (df['phobius-Prediction'] != 'o') & \
                  (df['phobius-Prediction'] != 'i')
    pred_df = df[nopred_cond].copy(deep=True)

    #  --  selecting accessions with sp with and without tm --  #
    sp_df = pred_df[pred_df['phobius-SP'].str.contains('Y')].copy(deep=True)
    new_sp_df = apply_filter(sp_df, 'sp')

    #  --  selecting accession only containing tm --  #
    sp_list = new_sp_df['Accession'].to_list()
    tm_only_df = pred_df[~pred_df['Accession'].isin(sp_list)].copy(deep=True)
    new_tm_only_df = apply_filter(tm_only_df, 'tm')

    #   -- concatenating phobius results  --  #
    cat = pd.concat([sp_df, new_tm_only_df])
    merged = pd.merge(df, cat, on=['Accession', 'phobius-TM', 'phobius-SP',
                                   'phobius-Prediction'], how='outer')
    del merged['tmp']
    return merged


def tmhmm_processing(df):
    df.replace(['ExpAA=', 'First60=', 'PredHel=', 'Topology='], '',
               regex=True, inplace=True)
    df = df.astype({'tmhmm-PredHel': 'int'})
    conditions = [(df['tmhmm-PredHel'] > 1),
                  (df['tmhmm-PredHel'] == 1),
                  (df['tmhmm-PredHel'] == 0)]
    choices = ['Multiple', 'Single', 'None']
    df['tm'] = np.select(conditions, choices, 'ERROR')
    return df


def add_tm_total(df):
    ndf = df.copy(deep=True)
    if 'DeepTMHMM.predicted.TMs' not in ndf.columns.to_list():
        ndf['DeepTMHMM.predicted.TMs'] = ndf['deepTMhelix'].apply(lambda x:
                                         len(x) if isinstance(x, list) else 0)
        return ndf
    else:
        return df


def guessing_headers(file_in):
    names = ['Accession', 'phobius-TM', 'phobius-SP', 'phobius-Prediction']
    ndf = pd.read_csv(file_in, sep=r'[ ]{1,}', header=0,
                      engine='python', names=names)
    ndf.dropna(axis=1, how='all', inplace=True)
    clean_df = parse_phobius(ndf)
    return clean_df


def sequence_properties(in_file, out_name, cmd):
    _ = lopit_utils.command_line_out('protein_features', **cmd)

    df = pd.read_csv(in_file, sep=r'\t|\,', header=0,
                     engine='python', na_values=['NA', ''])

    dic = dict(zip(df.Type, df.Path))
    all_dfs = []
    for key in dic.keys():
        if os.path.isfile(dic[key]):
            if key == 'fasta':
                f = read_fasta(dic[key])
            else:
                if key == 'phobius':
                    f = guessing_headers(dic[key])
                elif key == 'tmhmm':
                    f = tmhmm_processing(df)
                elif key == 'deeptmhmm':
                    pf = pd.read_csv(dic['deeptmhmm'], sep='\t', header=0)
                    f = add_tm_total(pf)
                elif key == 'deeploc':
                    tmp = pd.read_csv(dic[key], sep=r'\t|,', header=0)
                    c = {'Protein_ID': 'Accession',
                         'Localizations': 'deeploc.localizations'}
                    tmp.rename(columns=c, inplace=True)
                    f = tmp.loc[:, ['Accession', 'deeploc.localizations']]
                else:
                    f = pd.read_csv(dic[key], sep='\t', comment='#',
                                    names=headers[key])
            all_dfs.append(f)
        else:
            f = dic[key]
            print(f'unable to locate {f}.')
            sys.exit(-1)

    df_merged = reduce(lambda left, right: pd.merge(left, right,
                                                    on=['Accession'],
                                                    how='outer'), all_dfs)
    df_merged = df_merged[~df_merged['Accession'].str.contains('SEQENCE')]
    formatted_df = charm_prep(df_merged, out_name)
    return formatted_df


def read_fasta(in_file):
    dic = {}
    with open(in_file) as I:
        data = I.read().split('>')[1:]
        for sequence in data:
            i = sequence.split('\n')
            accession = i[0].split(' ')[0]
            header = i[0].replace(accession, '')
            dic[accession] = header
    df = pd.DataFrame.from_dict(dic, orient='index',
                                columns=['Annotation'])
    df.index.name = 'Accession'
    df.reset_index(inplace=True)
    return df


def charm_prep(df, out):
    df['SP_ranges'] = pd.cut(df['SP(Sec-SPI)'], bins=4,
                             labels=['0-0.25', '0.26-0.50',
                                     '0.51-0.75', '0,76-1.0'])
    df['TP_ranges'] = pd.cut(df['targetp-SP'], bins=4,
                             labels=['0-0.25', '0.26-0.50',
                                     '0.51-0.75', '0,76-1.0'])
    if 'gpi-anchored' in df.columns.to_list():
        df['gpi-anchored'] = df.loc[
            df['GPI-Anchored-pred'] == 'GPI-Anchored', 'gpi-Likelihood']
        df['gpi-anchored'].fillna(0, inplace=True)
    df.to_csv(f'{out}_formatted_protein_features.tsv', sep='\t', index=False)
    return df


#   ---  Execute   ---   #

if __name__ == '__main__':
    file_in = sys.argv[1]  # 'Sequence_properties.tsv => keys + file paths)
    outname = sys.argv[2]
    if os.path.isfile(file_in):
        charms_path = os.path.split(file_in)[0]
        os.chdir(charms_path)
        args = 'dictionary with all arguments'
        features_df = sequence_properties(file_in, outname, args)
    os.chdir('..')
    



