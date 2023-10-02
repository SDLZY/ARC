import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('val_stats_epoch_19_20211214.csv', index_col=0)
    df['shortcut'] = df['loss_1'] - df['loss_0']
    fig, ax = plt.subplots(1, 1, dpi=300)
    df.plot.scatter(x='gt_prob_1', y='gt_prob_2', s=0.1, ax=ax)
    # ax.set_ylim(-1, 1)
    plt.show()
    print(df)



if __name__ == '__main__':
    main()