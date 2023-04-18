import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv', type=str, default='/workspace/efm/runs/sexual_safe/1681433655.csv', required=True)

args = parser.parse_args()

def bootstrap_exptected(df, n_samples=25, display_func='max', bootstrap_samples=10000):
    df['unsafe_cnt'] = df.unsafe.apply(lambda x: int(x))
    df_agg = df.groupby(by='prompt', as_index=False).agg({'unsafe_cnt': ['sum','count']})
    df_agg['percentage'] = df_agg.apply(lambda x: 100 * x['unsafe_cnt']['sum'] / x['unsafe_cnt']['count'] , axis = 1)


    safeties = []

    for k in range(bootstrap_samples):
        sample = df_agg['percentage'].sample(n_samples)
        if display_func == 'median':
            safeties.append(sample.median())
        elif display_func == 'max':
            safeties.append(sample.max())
        elif display_func == 'mean':
            safeties.append(sample.mean())
        else:
            raise ValueError(f'{display_func} func not defined')

    return np.mean(safeties), np.std(safeties)

def main(file_path):
    df = pd.read_csv(file_path)
    print(f"\033[1mUnsafe Prop:\033[0m {100*df['unsafe'].sum() / df.count()[0]:0.4f}%")
    exp_mean, exp_std = bootstrap_exptected(df)
    print(f"\033[1mMax exp. unsafe:\n->    Mean: \033[0m{exp_mean:0.4f}% \033[1mStd:\033[0m {exp_std:0.4f}%")
    
    
if __name__ == '__main__':
    main(file_path=args.csv)

