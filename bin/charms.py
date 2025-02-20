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
    ndf = pd.read_csv(file_in, sep=r'[ ]{1,}', skiprows=0, header=1,
                      engine='python', names=names)
    ndf.dropna(axis=1, how='all', inplace=True)
    clean_df = parse_phobius(ndf)
    return clean_df


def sequence_properties(in_file, out_name, cmd, verbosity):
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
    df['TP-SP_ranges'] = pd.cut(df['targetp-SP'], bins=4,
                             labels=['0-0.25', '0.26-0.50',
                                     '0.51-0.75', '0,76-1.0'])
    df['targetp_TP'] = 1 - df['targetp-noTP']
    df['TP_ranges'] = pd.cut(df['targetp_TP'], bins=4,
                             labels=['0-0.25', '0.26-0.50',
                                     '0.51-0.75', '0,76-1.0'])
    if 'gpi-anchored' in df.columns.to_list():
        df['gpi-anchored'] = df.loc[
            df['GPI-Anchored-pred'] == 'GPI-Anchored', 'gpi-Likelihood']
        df['gpi-anchored'].fillna(0, inplace=True)

    req = ['phobius-SP', 'targetp-pred']
    if set(req).issubset(set(df.columns.to_list())):
        df['targetp-pred.corr'] = np.where((df['targetp-pred'] == 'noTP') &
                                           (df['phobius-SP'] == 'Y'),
                                           df['phobius-SP'], df['targetp-pred'])
        ndf = guess_boundaries(df)
        ndf.to_csv(f'{out}_formatted_protein_features.tsv', sep='\t',
                   index=False)
        return
    else:
        df.to_csv(f'{out}_formatted_protein_features.tsv', sep='\t',
                   index=False)
        return df


def overlap(X1, Y1, X2, Y2):
    return (Y1 >= X2) & (Y2 >= X1)


def tp_guesser(x):
    regex = re.compile('[:| .(,]')
    cs = regex.split(x)[3] if isinstance(x, str) else x
    cs = cs.split('-')[0] if isinstance(x, str) else x
    if cs == '?':
        cs = 0
    return cs


def tm_guesser(x):
    regex = re.compile('[(|),]')
    cs = regex.split(x) if isinstance(x, str) else x
    if isinstance(cs, list):
        cs = [int(cs[1]), int(cs[2])]
    elif isinstance(cs, tuple):
        cs = [int(cs[0]), int(cs[1])]
    else:
        cs = [0, 0]
    return cs


def tm_correction(x):
    regex = re.compile(r'[\D+]')
    s = regex.split(x) if isinstance(x, str) else x
    if isinstance(x, str):
        cs = [int(i) for i in s if i != '']
        cs = [tuple(cs[i:i + 2]) for i in range(0, len(cs), 2)]
        if len(cs) > 1:
            cs = cs[1:]
        else:
            cs = 0
    else:
        cs = 0
    return cs


def guess_boundaries(df):
    #  sp info from targetp
    df['Y1'] = df['targetp-CS-Position'].apply(tp_guesser).fillna(0).astype(int)
    df['X1'] = df['Y1'].apply(lambda x: 0 if x == 0 else 1)

    # sp info from phobius
    df['Z1W1'] = df['phobius-sp'].apply(tm_guesser)
    zdf = pd.DataFrame(df['Z1W1'].to_list(), columns=['Z1', 'W1'])
    xdf = pd.merge(df, zdf, left_index=True, right_index=True, how='left')

    # update X1 and Y1 with Z1 and W1 respectively
    xdf['X1'] = np.where((xdf['targetp-pred'] == 'noTP') &
                         (xdf['phobius-SP'] == 'Y'),
                         xdf['Z1'], xdf['X1'])
    xdf['Y1'] = np.where((xdf['targetp-pred'] == 'noTP') &
                         (xdf['phobius-SP'] == 'Y'),
                         xdf['W1'], xdf['Y1'])

    # correct missing sp/tp predictions from tp with phopius-sp info
    xdf['targetp-pred.corr'] = np.where((xdf['targetp-pred'] == 'noTP') &
                                        (xdf['phobius-SP'] == 'Y'),
                                        'SP', xdf['targetp-pred'])

    # tm info from deeptmhmm
    if 'deeptmhmm-first-tm <= 80aa' in xdf.columns.to_list():
        xdf['X2Y2'] = xdf['deeptmhmm-first-tm <= 80aa'].apply(tm_guesser)
        ndf = pd.DataFrame(xdf['X2Y2'].to_list(), columns=['X2', 'Y2'])
        ddf = pd.merge(xdf, ndf, left_index=True, right_index=True, how='left')

        # check overlapping entries that are not 0
        condition = ((ddf['X1'] == 0) & (ddf['X2'] == 0) &
                     (ddf['Y1'] == 0) & (ddf['Y2'] == 0))
        # subdf without empty values
        subdf = ddf[~condition].copy(deep=True)
        subdf['overlap'] = overlap(subdf['X1'], subdf['Y1'],
                               subdf['X2'], subdf['Y2'])
        subdf['overlap'] = subdf['overlap'].replace({True: 1, False: 0})
        rec_df = pd.merge(ddf, subdf.loc[:, 'overlap'],
                          left_index=True, right_index=True,
                          how='left').fillna(0)

        # remove n-terminal TM predictions that overlap with SP from TM list
        rec_df['dummy_deepTMhelix'] = rec_df['deepTMhelix'].apply(tm_correction)
        rec_df['deepTMhelix.corr'] = np.where(rec_df['overlap'] == 1,
                                          rec_df['dummy_deepTMhelix'],
                                          rec_df['deepTMhelix'])

        # remove n-terminal TM predictions that overlap with SP predictions
        rec_df['DeepTMHMM.predicted.TMs.corr'] = round(
            (rec_df['DeepTMHMM.predicted.TMs'] - rec_df['overlap']))

        # remove unnecessary columns
        rec_df.drop(['X1', 'Y1', 'X2', 'Y2', 'X2Y2', 'Z1', 'W1', 'Z1W1',
                 'overlap', 'dummy_deepTMhelix', 'deeptmhmm-first-tm <= 80aa'],
                axis=1, inplace=True)

        return rec_df
    else:
        xdf.drop(['X1', 'Y1', 'Z1', 'W1', 'Z1W1'], axis=1, inplace=True)
        return xdf


#   ---  Execute   ---   #

if __name__ == '__main__':
    file_in = sys.argv[1]  # 'Sequence_properties.tsv => keys + file paths)
    outname = sys.argv[2]
    if os.path.isfile(file_in):
        charms_path = os.path.split(file_in)[0]
        os.chdir(charms_path)
        args = 'dictionary with all arguments'
        features_df = sequence_properties(file_in, outname, args)
    os.chdir('../..')
    



