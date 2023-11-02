import torch
import torch.nn as nn
import torch.nn.functional as F

def LinearQuantizeOut(x, k, alpha):                # only quantize >0 values (relu must be preceded)
    if k == 1:
        return torch.sign(x).add(1).div(2).mul(alpha)
    else:
        L = 2.**k - 1
        xdiv = x.div(alpha)
        xc = xdiv.clamp(min=0., max=1.)
        xq = xc.mul(L).round()
        xmul = xq.div(L).mul(alpha)
    return xmul

def LinearQuantizeW(x, k, max_val, min_val):       # asymetric quant
    delta = max_val - min_val
    L= 2 ** k - 1
    stepSize = delta / L
    index = torch.clamp(torch.round((x-min_val) / delta * L), 0, L)
    return index, index * stepSize + min_val, stepSize
    
def clipping(weightQ, subArray, option):
    if option == 0:                 # normal mode
        threshold = subArray
    elif option == 1:               # Mean of weightQ
        threshold = weightQ.mean()
        # print(threshold)
    # elif option == 2:               # Clipping based-on output distribution with a  training dataset (mean + var)
    #     threshold = 
    # elif option == 3:               # Mean of output max value with a training dataset
    #     threshold = 
    else:                           # ideal point 
        threshold = 0

    return threshold

# Ideally, same to inputQ * weightQ if ADC=32bits
def MAC(inputQ, weightQ, abits, wbits, adcbits, output_size, subArray, clip_option=0):
    outputreal = F.linear(inputQ, weightQ, None)
    outputShiftIN = torch.zeros_like(outputreal)
    
    output_dist = []
    for z in range(abits):
        inputB = torch.fmod(inputQ, 2)              # 12,10,14 = [0,0,0] / [0,1,1] / [1,0,1] / [1,1,1]
        inputQ = torch.round((inputQ-inputB)/2)     # 12,10,14 = [6,5,7] / [3,2,3] / [1,1,1] / [0,0,0]
        weightQb = weightQ
        outputShiftW = torch.zeros_like(outputreal)
        for k in range(wbits):
            weightB = torch.fmod(weightQb, 2)
            weightQb = torch.round((weightQb-weightB)/2)
            threshold = clipping(weightQ, subArray, clip_option)
            
            outputPartial = F.linear(inputB, weightB, None)
            output_dist.append(outputPartial)
            outputPartial = torch.clamp(outputPartial, 0, threshold)
            outputADC = LinearQuantizeOut(outputPartial, adcbits, threshold)
            # shift per w bit sequence
            outputShiftW = outputShiftW + outputADC * (2 ** k)
        # shift per input bit sequence
        outputShiftIN = outputShiftIN + outputShiftW * (2 ** z)
    output_dist = torch.stack(output_dist)
    out_mean = output_dist.mean()
    out_max = output_dist.max()
    quant_err = (outputShiftIN - outputreal).abs().sum().div(outputADC.nelement()).item()
    # since inputQ [0, 15] when k=4, rescale output by divide 16
    # output = output + inputS * (outputIN * wS + outputIND * w.min())     # suppose I=[0~15], W=[-8~7] -> I*W = I[0~15]*W[0~15] + I[0~15]*W_constant[-8] (which is w.min())
    if output_size != outputShiftIN.size():
        outputShiftIN = outputShiftIN.transpose(1,2).reshape(output_size)
    return outputShiftIN, quant_err, out_mean.item(), out_max.item()


# Ideally, same to inputQ * weightQ if ADC=32bits
def CAM(inputQ, weightQ, abits, wbits, adcbits, output_size, subArray, s=0):
    # print(f'real: {F.linear(inputQ, weightQi, None)}')
    outputreal = F.linear(inputQ, weightQ, None)
    outputShiftIN = torch.zeros_like(outputreal)
    
    uniqs, cnts = torch.unique(weightQ, return_counts=True)
    for z in range(abits):
        inputB = torch.fmod(inputQ, 2)              # 12,10,14 = [0,0,0] / [0,1,1] / [1,0,1] / [1,1,1]
        inputQ = torch.round((inputQ-inputB)/2)     # 12,10,14 = [6,5,7] / [3,2,3] / [1,1,1] / [0,0,0]
        uniqs, cnts = torch.unique(weightQ, dim=1, return_counts=True)
        outputShiftW = torch.zeros_like(outputreal)
        # outputShiftW = F.linear(inputB, weightQ, None)  # approximately ~16.5s
        for un in uniqs: 
            maskCAM = (weightQ == un).float()
            outML = F.linear(inputB, maskCAM, None)
            # outMLADC = LinearQuantizeOut(outML, adcbits, subArray)
            outMLADC = outML # (original is of course better)
            outputShiftW = outputShiftW + outMLADC * un
        outputShiftIN = outputShiftIN + outputShiftW * (2 ** z)
    
    if output_size != outputShiftIN.size():
        outputShiftIN = outputShiftIN.transpose(1,2).reshape(output_size)
    return outputShiftIN, len(uniqs), cnts.max().item()


# Ideally, same to inputQ * weightQ if ADC=32bits
def ZP_MAC(inputQ, weightQ, abits, adcbits, output_size, subArray, zero_point_opt):
    outputreal = F.linear(inputQ, weightQ, None)
    outputDummyShift = torch.zeros_like(outputreal)
    if zero_point_opt:
        outputDummyShift = F.linear(inputQ, weightQ, None)
    else:
        for z in range(abits):
            inputB = torch.fmod(inputQ, 2)              
            inputQ = torch.round((inputQ-inputB)/2)     
            outputDummy = F.linear(inputB, weightQ, None)
            outputDummyADC = LinearQuantizeOut(outputDummy, adcbits, subArray)
            outputDummyShift = outputDummyShift + outputDummyADC * (2 ** z)
    # since inputQ [0, 15] when k=4, rescale output by divide 16
    # output = output + inputS * (outputIN * wS + outputIND * w.min())     # suppose I=[0~15], W=[-8~7] -> I*W = I[0~15]*W[0~15] + I[0~15]*W_constant[-8] (which is w.min())
    if output_size != outputDummyShift.size():
        outputDummyShift = outputDummyShift.transpose(1,2).reshape(output_size)
    return outputDummyShift

# default: vanilla
# inference = 0: real (FP32)
# inference = 1: Activation/Weight Quantize
# inference = 2: Activation/Weight/Output Quantize
# inference = 3: PIM (Activation/Weight/Output Quantize)
# inference = -1: PIM-mimic (where consider dummy)
class QConv2dCAM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bitActivation=8,bitWeight=[4,4],
                 inference=0,subArray=128,bitADC=5,ADClamp=1,zero_point_opt=False):
        super(QConv2dCAM, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert isinstance(bitWeight, list)
        self.bitWeight = bitWeight
        self.bitWeightMSB, self.bitWeightLSB = bitWeight
        self.bitActivation = bitActivation
        self.inference = inference
        self.subArray = subArray
        self.bitADC = bitADC
        self.ADClamp = ADClamp
        self.zero_point_opt = zero_point_opt
        self.quant_err = {}
        self.out_mean = {}
        self.out_max = {}
        self.search_time = {}
        self.search_cnt = {}
        
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', ibits={bitActivation}, wbits={bitWeight}, inference={inference}, subArray={subArray}, ADCbits={bitADC}'
        return s.format(**self.__dict__)
        
    # convert input -> # [N, OH * OW, IC * KH * KW]
    def input_2d_mapping(self, input):
        fold_param = dict(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_module = nn.Unfold(**fold_param)
        unfold_out = unfold_module(input)
        return unfold_out.transpose(1,2)
    
    # convert: weight -> # [IC * KH * KW, OC]
    def weight_2d_mapping(self, weight):
        return weight.reshape(weight.shape[0], -1)
    
    def __data_update__(self, index, search_time, search_cnt, quant_err, out_mean, out_max):
        self.search_time[f"{self.subArray}_{index}"] = search_time
        self.search_cnt[f"{self.subArray}_{index}"] = search_cnt
        self.quant_err[f"{self.subArray}_{index}"] = quant_err
        self.out_mean[f"{self.subArray}_{index}"] = out_mean
        self.out_max[f"{self.subArray}_{index}"] = out_max
        
    def forward(self, input):
        # make input & weight to 2D tensors
        input2D = self.input_2d_mapping(input)                                                           
        weight2D = self.weight_2d_mapping(self.weight)                              

        inputQ, inputQS, inputS = LinearQuantizeW(input2D, self.bitActivation, input2D.max(), input2D.min())
        weightQ, weightQS, weightS = LinearQuantizeW(weight2D, sum(self.bitWeight), weight2D.max(), weight2D.min())
        
        outputreal = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        weightQM = weightQ // (2.**self.bitWeightLSB)
        if self.bitWeightMSB == 0:                              # if MSB=0bit, then use weightQ
            weightQL = weightQ
        else:
            if self.bitWeightLSB == 0:
                weightQL = weightQ // (2.**self.bitWeightLSB)
            else:
                weightQL = weightQ % (2.**self.bitWeightLSB)
        
        numSubArray = int(weight2D.shape[1]/self.subArray)
        if self.inference == 3:
            if numSubArray == 0:
                outputM, search_time, search_cnt = CAM(inputQ, weightQM, self.bitActivation, self.bitWeightMSB, self.bitADC, outputreal.size(), weight2D.shape[1])
                outputL, quant_err, out_mean, out_max = MAC(inputQ, weightQL, self.bitActivation, self.bitWeightLSB, self.bitADC, outputreal.size(), weight2D.shape[1], self.ADClamp)
                outputDL = ZP_MAC(inputQ, torch.ones_like(weightQ), self.bitActivation, self.bitADC, outputreal.size(), weight2D.shape[1], self.zero_point_opt)
                self.__data_update__(self.subArray, search_time, search_cnt, quant_err, out_mean, out_max)
                
                outputP = (outputM * (2.**self.bitWeightLSB) + outputL) * weightS
                outputD = outputDL * weight2D.min()
                out = inputS * (outputP + outputD)
            else:
                numSubRow = [self.subArray] * numSubArray + ([] if weight2D.shape[1] % self.subArray == 0 else [weight2D.shape[1] % self.subArray])
                out = torch.zeros_like(outputreal)
                index = 0
                for s, rowArray in enumerate(numSubRow):
                    index += rowArray
                    mask = torch.zeros_like(weight2D)
                    mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                    outputM, search_time, search_cnt = CAM(inputQ, weightQM*mask, self.bitActivation, self.bitWeightMSB, self.bitADC, outputreal.size(), rowArray, s)
                    outputL, quant_err, out_mean, out_max = MAC(inputQ, weightQL*mask, self.bitActivation, self.bitWeightLSB, self.bitADC, outputreal.size(), rowArray, self.ADClamp)
                    outputDL = ZP_MAC(inputQ, torch.ones_like(weightQ)*mask, self.bitActivation, self.bitADC, outputreal.size(), rowArray, self.zero_point_opt)
                    self.__data_update__(index, search_time, search_cnt, quant_err, out_mean, out_max)
                    
                    outputP = (outputM * (2.**self.bitWeightLSB) + outputL) * weightS
                    outputD = outputDL * weight2D.min()
                    out = out + inputS * (outputP + outputD)
                
            if self.bias is not None:
                out = out + self.bias
        elif self.inference == 2:
            output = F.linear(inputQS, weightQS, self.bias)
            _, out, _ = LinearQuantizeW(output, self.bitADC, output.max(), output.min())
        elif self.inference == 1:
            out = F.linear(inputQS, weightQS, self.bias)
        elif self.inference == -1:
            outputM = F.linear(inputQ, weightQM, None).transpose(1,2).reshape(outputreal.size())
            outputL = F.linear(inputQ, weightQL, None).transpose(1,2).reshape(outputreal.size())
            outputDL = F.linear(inputQ, torch.ones_like(weightQ), None).transpose(1,2).reshape(outputreal.size())
            
            outputP = (outputM * (2.**self.bitWeightLSB) + outputL) * weightS
            outputD = outputDL * weight2D.min()
            out = inputS * (outputP + outputD)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = outputreal
        return out
    
# default: vanilla
# inference = 0: real (FP32)
# inference = 1: Activation/Weight Quantize
# inference = 2: Activation/Weight/Output Quantize
# inference = 3: PIM (Activation/Weight/Output Quantize)
class QLinearCAM(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, bitActivation=8,bitWeight=[4,4],inference=0,subArray=128,bitADC=5,zero_point_opt=False):
        super(QLinearCAM, self).__init__(in_features, out_features, bias)
        self.bitWeight = bitWeight
        self.bitWeightMSB, self.bitWeightLSB = bitWeight
        self.bitActivation = bitActivation
        self.inference = inference
        self.subArray = subArray
        self.bitADC = bitADC
        self.zero_point_opt = zero_point_opt

    def extra_repr(self):
        s = ('{in_features}, {out_features}')
        if self.bias is None:
            s += ', bias=False'
        s += ', ibits={bitActivation}, wbits={bitWeight}, inference={inference}, subArray={subArray}, ADCbits={bitADC}'
        return s.format(**self.__dict__)
        
    def forward(self, input):
        inputQ, inputQS, inputS = LinearQuantizeW(input, self.bitActivation, input.max(), input.min())
        weightQ, weightQS, weightS = LinearQuantizeW(self.weight, sum(self.bitWeight), self.weight.max(), self.weight.min())
        
        outputreal = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        weightQM = weightQ // (2.**self.bitWeightLSB)
        weightQL = weightQ % (2.**self.bitWeightLSB)
        
        numSubArray = int(self.weight.shape[1]/self.subArray)
        if self.inference == 3:
            if numSubArray == 0:
                outputM = CAM(inputQ, weightQM, self.bitActivation, self.bitWeightMSB, self.bitADC, outputreal.size(), self.weight.shape[1])
                outputL, outputDL = MAC(inputQ, weightQL, self.bitActivation, self.bitWeightLSB, self.bitADC, outputreal.size(), self.weight.shape[1])

                outputP = (outputM * (2.**self.bitWeightLSB) + outputL) * weightS
                if self.zero_point_opt:
                    outputD = F.linear(inputQ, torch.ones_like(self.weight), None).transpose(1,2).reshape(outputreal.shape) * self.weight.min()
                else:
                    outputD = outputDL * self.weight.min()
                out = inputS * (outputP + outputD)
            else:
                numSubRow = [self.subArray] * numSubArray + ([] if self.weight.shape[1] % self.subArray == 0 else [self.weight.shape[1] % self.subArray])
                out = torch.zeros_like(outputreal)
                outputM = torch.zeros_like(outputreal)
                outputL = torch.zeros_like(outputreal)
                for s, rowArray in enumerate(numSubRow):
                    mask = torch.zeros_like(self.weight)
                    mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                    
                    Sub_outputM = CAM(inputQ, weightQM*mask, self.bitActivation, self.bitWeightMSB, self.bitADC, outputreal.size(), rowArray)
                    Sub_outputL, outputDL = MAC(inputQ, weightQL, self.bitActivation, self.bitWeightLSB, self.bitADC, outputreal.size(), rowArray, mask)
                    outputM = outputM + Sub_outputM
                    outputL = outputL + Sub_outputL
                outputP = (outputM * (2.**self.bitWeightLSB) + outputL) * weightS
                if self.zero_point_opt:
                    outputD = F.linear(inputQ, torch.ones_like(self.weight), None).transpose(1,2).reshape(outputreal.shape) * self.weight.min()
                else:
                    outputD = outputDL * self.weight.min()
                out = inputS * (outputP + outputD)
            if self.bias is not None:
                out = out + self.bias
        elif self.inference == 2:
            output = F.linear(inputQS, weightQS, self.bias)
            _, out, _ = LinearQuantizeW(output, self.bitADC, output.max(), output.min())
        elif self.inference == 1:
            out = F.linear(inputQS, weightQS, self.bias)
        else:
            out = outputreal
        return out
    
    
