import json
import os
import time
import argparse
import simulation_KT as KT
from param import KTParam
from dataset import KTInitDataSet
from value import ValueTrainer
from policy import KTPolicyTrainer
from util import print_elapsedtime

# argparse を使ってコマンドライン引数を定義
parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--config_path', type=str, default='game_nn_n50_0fm1gm.json', help='The path to load json file.')
parser.add_argument('--exp_name', type=str, default='test', help='The suffix used in model_path for save.')
parser.add_argument('--save_files', action='store_true', help='If set, files will be saved.')

KT.seed_everything(42)
# コマンドライン引数を解析
args = parser.parse_args()
def main():
    # 解析した引数にアクセス
    config_path = args.config_path
    exp_name = args.exp_name
    save_files = args.save_files

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Solving the problem based on the config path {}".format(config_path))
    mparam = KTParam(config["n_agt"], config["beta"], config["mats_path"])
    
    base_model_path = r"C:\Users\Owner\OneDrive\デスクトップ\Github\Khan-and-Thomas\results"
    model_path = os.path.join(base_model_path, "{}_{}_n{}_{}".format(
        "game" if config["policy_config"]["opt_type"] == "game" else "sp",
        config["dataset_config"]["value_sampling"],
        config["n_agt"],
        args.exp_name,  # FLAGS.exp_name の代わりに args.exp_name を使用
    ))
    
    if save_files:
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, "config_beg.json"), 'w') as f:
            json.dump(config, f)
    
    start_time = time.time()
    init_ds = KTInitDataSet(mparam, config)
    value_config = config["value_config"]
    if config["init_with_bchmk"]:
        init_policy = init_ds.k_policy_bchmk
        policy_type = "pde"
        # TODO: change all "pde" to "conventional"
    else:
        init_policy = init_ds.c_policy_const_share
        policy_type = "nn_share"
        
    vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
    policy_config = config["policy_config"]
    price_config = config["price_config"]
    ptrainer = KTPolicyTrainer(vtrainers, init_ds)
    train_vds, valid_vds = init_ds.get_valuedataset(init_ds.policy_init_only, "nn_share", ptrainer.price_fn, init=True, update_init=False)
    
    for vtr in vtrainers:
        vtr.train(train_vds, valid_vds, 100, value_config["batch_size"])
    
    plt = ptrainer.train(60, policy_config["batch_size"])
    plt.show()
    
    if save_files:
        with open(os.path.join(model_path, "config.json"), 'w') as f:
            json.dump(config, f)
        for i, vtr in enumerate(vtrainers):
            vtr.save_model(os.path.join(model_path, "value{}.pth".format(i)))
        ptrainer.save_model(os.path.join(model_path, "policy.pth"))

    end_time = time.monotonic()
    print_elapsedtime(end_time - start_time)

if __name__ == '__main__':
    main()