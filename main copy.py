import copy
import random
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

import torch
import copy
import torch.nn as nn
import itertools
import pandas as pd

from qlayercam import QConv2dCAM
from qlayer import QLinear, QConv2d
from utils.config import FLAGS
from utils.dataloader import *

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def set_deep_attr(obj, attrs, value):
    for attr in attrs.split(".")[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs.split(".")[-1], value)
    
def load_model(model_name):
    if model_name == 'resnet20':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True) # pretrained cifar10 model load
    return model

def change_model(model, args):
    copy_model = copy.deepcopy(model)
    hid_abits, hid_wbits, hid_inference, hid_subarray, hid_adcbits, hid_adclamp, hid_zero_point_opt = args
    
    total_layer_num = len(list(model.named_modules()))
    for i, (name, m) in enumerate(model.named_modules()): # downsample no quantize !
        if getattr(m, "in_channels", False) == 3 or i == (total_layer_num - 1):
            model_configs = [FLAGS.fl_abits, FLAGS.fl_wbits, FLAGS.fl_inference, FLAGS.fl_subarray, FLAGS.fl_adcbits]
        else:
            model_configs = [hid_abits, hid_wbits, hid_inference, hid_subarray, hid_adcbits, hid_adclamp, hid_zero_point_opt]
        if isinstance(m, nn.Conv2d) and 'downsample' not in name:
            F = QConv2dCAM if m.in_channels != 3 and getattr(FLAGS, "cam", False) else QConv2d

            set_deep_attr(copy_model, name, F(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, m.bias, *model_configs))
            for p_name, p in m.named_parameters():
                set_deep_attr(copy_model, name + '.' + p_name, p)
        elif isinstance(m, nn.Linear):
            # F = QLinear if i == (total_layer_num - 1) else QLinearCAM
            F = QLinear
            
            set_deep_attr(copy_model, name, QLinear(m.in_features, m.out_features, True if m.bias is not None else False, *model_configs))
            for p_name, p in m.named_parameters():
                set_deep_attr(copy_model, name + '.' + p_name, p)
    # print(copy_model)
    return copy_model


def verify(model):
    for name, m in model.named_modules():
        if hasattr(m, "inputQ") or hasattr(m, "weightQ"):
            try:
                print(name, len(m.inputQ), len(m.weightQ))
            except:
                pass
            # if len(m.inputQ) > 16:
            #     print(m.inputQ)

def cartesian_product(*args):
    args_list = []
    for val in list(args):
        if isinstance(val, (int, float)):
            args_list.append([val])
        else:
            args_list.append(val)
    return itertools.product(*args_list)


from qlayercam import LinearQuantizeW
class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        # self.input = input
        self.weight = module.weight
        # self.output = output
        
        if hasattr(module, "bitWeight"):
            weight2D = module.weight_2d_mapping(self.weight)
            weightQ, weightQS, weightS = LinearQuantizeW(weight2D, sum(self.bitWeight), weight2D.max(), weight2D.min())
        
        weightQM = weightQ // (2.**module.bitWeightLSB)
        weightQL = weightQ % (2.**module.bitWeightLSB)
        
        # self.approx_weight = quantize(self.weight, module.step_size, module.half_lvls) * module.step_size
        # self.approx_input = quantize(self.input[0], self.input_step_size, module.full_lvls -1) * self.input_step_size
        # self.approx_output = F.conv2d(self.approx_input, self.approx_weight, module.bias, module.stride,
        #                     module.padding, module.dilation, module.groups)
    def close(self):
        self.hook.remove()

def plot(name, df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Month', fontsize=14)
    sns.heatmap(df, cmap='YlOrBr')
    plt.show()
    plt.savefig(name)


def main():
    @torch.inference_mode()
    def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        extra_preprocess = None
    ) -> float:
        model.eval()
        model.cuda()
        num_samples = 0
        num_correct = 0
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="eval", leave=False)):
            # Move the data from CPU to GPU
            inputs = inputs.cuda()
            if extra_preprocess is not None:
                for preprocess in extra_preprocess:
                    inputs = preprocess(inputs)
            targets = targets.cuda()
            outputs = model(inputs)
            # Convert logits to class indices
            outputs = outputs.argmax(dim=1)
            # Update metrics
            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()
            if i == 0:
                break
        return (num_correct / num_samples * 100).item()

    dataLoader = load_dataset(FLAGS.dataset, FLAGS.num_samples)
    configuration = cartesian_product(FLAGS.hid_abits, FLAGS.hid_wbits, FLAGS.hid_inference, FLAGS.hid_subarray, FLAGS.hid_adcbits, FLAGS.hid_adclamp, FLAGS.hid_zero_point_opt)
    
    c_model = []
    model = load_model(FLAGS.model)
    final_result = {}
    
    for config in configuration:
        config_name = "_".join([str(i) for i in config])
        print(f"CONFIG: {config}")
        c_model = change_model(model, config)
    
        if getattr(FLAGS, 'clip_mode', 0) > 1:      # use training dataset for calibration
            ptq_int8_model_accuracy = evaluate(c_model, dataLoader['test'])    
        else:
            _ = evaluate(c_model, dataLoader['train'])
            
            ptq_int8_model_accuracy = evaluate(c_model, dataLoader['test'])    
        
        print(f"ptq model has accuracy={ptq_int8_model_accuracy:.2f}%")

        quant_err, out_mean, out_max, search_time, search_cnt = [{} for _ in range(5)]
        idx = 0
        for _, m in c_model.named_modules():
            if hasattr(m, 'quant_err'):
                quant_err[f"L{idx}"] = m.quant_err
                out_mean[f"L{idx}"] = m.out_mean
                out_max[f"L{idx}"] = m.out_max
                search_time[f"L{idx}"] = m.search_time
                search_cnt[f"L{idx}"] = m.search_cnt
                idx += 1
                
        quant_err_df = pd.DataFrame(quant_err)
        out_mean_df = pd.DataFrame(out_mean)
        out_max_df = pd.DataFrame(out_max)
        search_time_df = pd.DataFrame(search_time)
        search_cnt_df = pd.DataFrame(search_cnt)
        summary = pd.DataFrame({'Accuracy': {k: ptq_int8_model_accuracy if k == 'L0' else None for k in quant_err.keys()},
                                'quant_err': quant_err_df.mean(),
                                'out_mean': out_mean_df.mean(),
                                'out_max': out_max_df.mean(),
                                'search_time': search_time_df.mean(),
                                'search_cnt': search_cnt_df.mean()})
        
        with pd.ExcelWriter(f"./result/{config_name}.xlsx") as writer:
            summary.to_excel(writer, sheet_name="summary", index=True)
            quant_err_df.transpose().to_excel(writer, sheet_name="quant_err", index=True)
            out_mean_df.transpose().to_excel(writer, sheet_name="out_mean", index=True)
            out_max_df.transpose().to_excel(writer, sheet_name="out_max", index=True)
            search_time_df.transpose().to_excel(writer, sheet_name="search_time", index=True)
            search_cnt_df.transpose().to_excel(writer, sheet_name="search_cnt", index=True)
        final_result[config_name] = {'Accuracy': ptq_int8_model_accuracy,
                                    'quant_err': quant_err_df.mean().mean(),
                                    'out_mean': out_mean_df.mean().mean(),
                                    'out_max': out_max_df.mean().mean()}
    fr = pd.DataFrame(final_result).transpose()
    fr.to_excel("result.xlsx")

if __name__ == "__main__":
    main()
    

