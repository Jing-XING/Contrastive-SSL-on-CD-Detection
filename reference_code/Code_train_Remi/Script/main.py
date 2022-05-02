from basic_trainer import Trainer
from config import get_config
from data_loader import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import time
import os
def main(config):
    cross_acc = 0

    cross_val_num = config.num_crossVal
    gen_cm = np.zeros((config.nb_classes, config.nb_classes))

    for Cval_num in range(cross_val_num):
        seed = config.random_seed + Cval_num
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        np.random.seed(seed)
        random.seed(seed)

        trainloaders = get_train_valid_loader_consensus(config.csv, batch_size=16,
                                                        random_seed=88,
                                                        subset='train',
                                                        num_workers=0,
                                                        mode=config.shuffle_mode, test_size=config.test_size, valid_size=config.valid_size)

        validloaders = get_train_valid_loader_consensus(config.csv, batch_size=16,
                                                        random_seed=88,
                                                        subset='valid',
                                                        num_workers=0,
                                                        mode=config.shuffle_mode, test_size=config.test_size, valid_size=config.valid_size)
        testloaders = get_train_valid_loader_consensus(config.csv, batch_size=16,
                                                        random_seed=88,
                                                        subset='test',
                                                        num_workers=0,
                                                        mode=config.shuffle_mode, test_size=config.test_size, valid_size=config.valid_size)
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        for i in range(len(trainloaders)):
            log_path = f'./log/{now}/subset{i}'
            os.makedirs(log_path)
            writer = SummaryWriter(log_path)
            trainer = Trainer(config, trainloaders[i], validloaders[i],
                                testloaders[i], iteration_value=5 * Cval_num + i, writer)
            trainer.train()
            torch.cuda.empty_cache()

        

      

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)