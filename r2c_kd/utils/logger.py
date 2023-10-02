import logging
import torch


class MeanLogger:
    # 静态变量，用字典储存和索引所有的logger
    logger_dict = dict()

    def __init__(self, name, period=((0, 100, 2000), (5, 100, 1000))):
        """

        Args:
            name:
            period: 类型int时每period次迭代统计并输出， 否则为[[x0, x1, ...], [p0, p1, ...]],
            表示在x_i次迭代后使用p_i作为period
        """
        self.period = period
        self.name = name
        self.step = 0
        self.total_value = 0
        self.total_counts = 0

    def clear(self):
        self.total_value = 0
        self.total_counts = 0

    def log(self, data):
        assert len(data.shape) == 1, f'data shape {data.shape} to log invalid'
        self.step += 1
        if isinstance(data, torch.Tensor):
            self.total_value += data.sum().item()
        else:
            self.total_value += data.sum()
        self.total_counts += data.shape[0]
        if self.check_period():
            mean_value = self.total_value / self.total_counts
            logging.info(f'[STEP: {self.step}] MEAN LOGGER[{self.name}]: {mean_value}')
            self.clear()

    def check_period(self):
        if isinstance(self.period, int):
            return self.step % self.period == 0
        else:
            for i, end in enumerate(self.period[0][::-1]):
                if self.step > end:
                    return self.step % self.period[1][i] == 0

    @staticmethod
    def get_logger(name):
        if name in MeanLogger.logger_dict:
            logger = MeanLogger.logger_dict[name]
        else:
            logger = MeanLogger(name)
            MeanLogger.logger_dict[name] = logger
        return logger
