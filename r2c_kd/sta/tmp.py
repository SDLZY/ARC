"""用tensorboard可视化baseline模型训练过程中instance-level各项数据的统计数据"""
import numpy as np
import pandas as pd
import os
import re
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


def plot_shortcut_by_correct2():
    root = '/home/share/wangkejie/vcrnb2/r2c/saves/my_model_stats2'
    epochs = [
        # pylint: disable=anomalous-backslash-in-string
        int(re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1))
        for x in os.listdir(root) if 'model_state_epoch' in x
    ]
    epochs.sort()
    print(f'Max epoch is {max(epochs)}')
    # writer = SummaryWriter(log_dir='loss1_loss0_baseline')
    fig, ax = plt.subplots(20, 4, dpi=128, figsize=(16, 40))
    for sidx, split in enumerate(('train', 'val')):
        df_arr = []
        for epoch in epochs:
            df = pd.read_csv(os.path.join(root, f'{split}_stats_epoch_{epoch}.csv'), index_col=0)
            df['epoch'] = epoch
            df_arr.append(df)
        # df = df_arr[23]
        for i in range(20):
            for tidx, (tag, group) in enumerate(df_arr[i].groupby(df_arr[i].correct_2)):
            # for tidx, (tag, group) in enumerate(df_arr[i].groupby(df_arr[i].correct_1)):
                print(f'{split} {i} qa2r: {tag}: {(group.loss_1 - group.loss_0>0).sum()}/{group.shape[0]}')
                # print(f'{split} {i} q2a: {tag}: {(group.loss_1 - group.loss_0>0).sum()}/{group.shape[0]}')
                (group.loss_1 - group.loss_0).hist(ax=ax[i, sidx * 2 + tidx], bins=100)
                ax[i, sidx * 2 + tidx].set_title(f'{split} {i} qa2r: {tag}')
                # ax[i, sidx * 2 + tidx].set_title(f'{split} {i} q2a: {tag}')
    plt.tight_layout()
    plt.savefig('loss_1-loss_0_baseline_cond_on_qa2r.png')
    plt.show()


def plot_shortcut_by_acc1():
    root = '/home/share/wangkejie/vcrnb2/r2c/saves/my_model_stats2'
    epochs = [
        # pylint: disable=anomalous-backslash-in-string
        int(re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1))
        for x in os.listdir(root) if 'model_state_epoch' in x
    ]
    epochs.sort()
    print(f'Max epoch is {max(epochs)}')
    # writer = SummaryWriter(log_dir='loss1_loss0_baseline')
    fig, ax = plt.subplots(2, 20, dpi=128, figsize=(80, 8), sharex=True, sharey=True)
    for sidx, split in enumerate(('train', 'val')):
        df_arr = []
        for epoch in epochs:
            df = pd.read_csv(os.path.join(root, f'{split}_stats_epoch_{epoch}.csv'), index_col=0)
            df['epoch'] = epoch
            df_arr.append(df)
        # df = df_arr[23]
        # sample_indexes = df_arr[0].sample(500).index
        for i in range(20):
            df = df_arr[i]
            df['shortcut'] = df_arr[i].loss_1 - df_arr[i].loss_0
            # df.loc[sample_indexes].plot.scatter(x='shortcut', y='gt_prob_1', s=0.5, ax=ax[sidx, i])
            df.plot.scatter(x='shortcut', y='gt_prob_1', s=0.5, ax=ax[sidx, i])
            ax[sidx, i].set_title(f'{split} {i}')
    plt.tight_layout()
    plt.savefig('loss_1-loss_0_acc1_baseline.png')
    plt.show()


def main():
    # plot_shortcut_by_acc1()
    plot_shortcut_by_correct2()


if __name__ == '__main__':
    main()
                # writer.add_histogram(tag=f'loss0/q2a_{tag}/{split}', values=group.loss_0.values, global_step=i)
                # writer.add_histogram(tag=f'loss1/q2a_{tag}/{split}', values=group.loss_1.values, global_step=i)
                # writer.add_histogram(tag=f'loss1-loss0/q2a_{tag}/{split}', values=(group.loss_1 - group.loss_0).values, global_step=i)
    # l0 = df.loss_0
    # l1 = df.loss_1
    # fig, ax = plt.subplots(2, 3, figsize=(18, 12), dpi=128)
    # sns.distplot(l0, kde=False, ax=ax[0,0])
    # sns.distplot(l1, kde=False, ax=ax[0,1])
    # sns.distplot(l1-l0, kde=False, ax=ax[0,2])
    # sns.distplot(l0, kde=False, ax=ax[1,0], bins=np.arange(0, 10, 0.2))
    # sns.distplot(l1, kde=False, ax=ax[1,1], bins=np.arange(0, 10, 0.2))
    # sns.distplot(l1-l0, kde=False, ax=ax[1,2], bins=np.arange(-10, 10, 0.2))
    # plt.savefig('fig_loss1_loss0.png')
    # plt.show()
     # df = pd.concat(df_arr)
    # 由于Dataloader drop last是True,每个epoch中参与训练集的样本不全一样

    # groups = df.groupby(['correct_1', 'epoch']).apply(lambda x: x.mean())

