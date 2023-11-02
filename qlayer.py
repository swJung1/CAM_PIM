import torch
import torch.nn as nn
import torch.nn.functional as F

def LinearQuantizeOut(x, k, alpha):                # only quantize >0 values (relu must be preceded)
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

# default: vanilla
# inference = 0: Quantize
# inference = 1: Activation/Weight Quantize
# inference = 2: Activation/Weight/Output Quantize
# inference = 3: Quantize + ADC
class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, wl_input=6,wl_weight=6,inference=1,subArray=128,ADCprecision=5,zero_point_opt=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.cellBit = 1                # SRAM
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.zero_point_opt = zero_point_opt

    def extra_repr(self):
        s = ('{in_features}, {out_features}')
        if self.bias is None:
            s += ', bias=False'
        s += ', ibits={wl_input}, wbits={wl_weight}, inference={inference}, subArray={subArray}, ADCbits={ADCprecision}'
        return s.format(**self.__dict__)

    def forward(self, input):
        outputOrignal = F.linear(input, self.weight, self.bias)
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        if self.inference == 3:
            # set parameters for Hardware Inference
            output = torch.zeros_like(outputOrignal)
            cellRange = 2 # 2**self.cellBit   # due to SRAM, it has only 2 cellRange
            # need to divide to different subArray
            numSubArray = int(self.weight.shape[1]/self.subArray)
            if numSubArray == 0:
                inputQ, _, inputS = LinearQuantizeW(input, bitActivation, input.max(), input.min())
                outputIN = torch.zeros_like(outputOrignal)
                outputIND = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)              # 12,10,14 = [0,0,0] / [0,1,1] / [1,0,1] / [1,1,1]
                    inputQ = torch.round((inputQ-inputB)/2)     # 12,10,14 = [6,5,7] / [3,2,3] / [1,1,1] / [0,0,0]
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    weightQ, _, weightS = LinearQuantizeW(self.weight, bitWeight, self.weight.max(), self.weight.min())      # [-1,1] -> [0,2] -> [0,1] by mod 2 -> (2^k-1) * weight [0,1] -> -1~1 mapped to 0~2^k-1
                    outputP = torch.zeros_like(outputOrignal)
                    for k in range (int(bitWeight/self.cellBit)):
                        weightB = torch.fmod(weightQ, cellRange)
                        weightQ = torch.round((weightQ-weightB)/cellRange)
                        # noise variation
                        # variation = np.random.normal(0, vari, list(weightQ.size())).astype(np.float32)
                        # weightQ = torch.round((weightQ-remainder)/cellRange)
                        outputPartial = F.linear(inputB, weightB, self.bias)
                        outputDummy = F.linear(inputB, torch.ones_like(weightB), self.bias)         # only 1-bit (all stored 1-logic)
                        # Add ADC quanization effects here !!!
                        outputADC = LinearQuantizeOut(outputPartial, self.ADCprecision, self.weight.shape[1])
                        outputDummyADC = LinearQuantizeOut(outputDummy, self.ADCprecision, self.weight.shape[1])
                        # shift per weight bit sequence
                        outputP = outputP + outputADC * (cellRange ** k)
                    # shift per input bit sequence
                    outputIN = outputIN + outputP * (2 ** z)
                    outputIND = outputIND + outputDummyADC * (2 ** z)                               # this is basically same to sigma(input)
                # since inputQ [0, 15] when k=4, rescale output by divide 16
                output = output + inputS * (outputIN * weightS + outputIND * self.weight.min())     # suppose I=[0~15], W=[-8~7] -> I*W = I[0~15]*W[0~15] + I[0~15]*W_constant[-8] (which is weight.min())
                # output = output + inputS * outputIN * weightS + outputIND * inputS * self.weight.min()
            else:
                outputF = torch.zeros_like(outputOrignal)
                numSubArray = self.weight.shape[1] // self.subArray
                numSubRow = [self.subArray] * numSubArray + ([] if self.weight.shape[1] % self.subArray == 0 else [self.weight.shape[1] % self.subArray])
                for s, rowArray in enumerate(numSubRow):
                    # different from neurosim / just initialize input again
                    inputQ, _, inputS = LinearQuantizeW(input, bitActivation, input.max(), input.min())
                    mask = torch.zeros_like(self.weight)
                    mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                    outputIN = torch.zeros_like(outputOrignal)
                    outputIND = torch.zeros_like(outputOrignal)
                    for z in range(bitActivation):
                        inputB = torch.fmod(inputQ, 2)              # 12,10,14 = [0,0,0] / [0,1,1] / [1,0,1] / [1,1,1]
                        inputQ = torch.round((inputQ-inputB)/2)     # 12,10,14 = [6,5,7] / [3,2,3] / [1,1,1] / [0,0,0]
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        weightQ, _, weightS = LinearQuantizeW(self.weight*mask, bitWeight, self.weight.max(), self.weight.min())      # [-1,1] -> [0,2] -> [0,1] by mod 2 -> (2^k-1) * weight [0,1] -> -1~1 mapped to 0~2^k-1
                        outputP = torch.zeros_like(outputOrignal)
                        for k in range (int(bitWeight/self.cellBit)):
                            weightB = torch.fmod(weightQ, cellRange)*mask
                            weightQ = torch.round((weightQ-weightB)/cellRange)*mask
                            # noise variation
                            # variation = np.random.normal(0, vari, list(weightQ.size())).astype(np.float32)
                            # weightQ = torch.round((weightQ-remainder)/cellRange)
                            outputPartial = F.linear(inputB, weightB, self.bias)
                            # Add ADC quanization effects here !!!
                            outputADC = LinearQuantizeOut(outputPartial, self.ADCprecision, rowArray)
                            # shift per weight bit sequence
                            outputP = outputP + outputADC * (cellRange ** k)
                        # only 1-bit (all stored 1-logic)
                        outputDummy = F.linear(inputB, torch.ones_like(weightB)*mask, self.bias)
                        outputDummyADC = LinearQuantizeOut(outputDummy, self.ADCprecision, rowArray)
                        # shift per input bit sequence
                        outputIN = outputIN + outputP * (2 ** z)
                        outputIND = outputIND + outputDummyADC * (2 ** z)                           # this is basically same to sigma(input)
                    # since inputQ [0, 15] when k=4, rescale output by divide 16
                    outputF = outputF + inputS * (outputIN * weightS + outputIND * self.weight.min())    # suppose I=[0~15], W=[-8~7] -> I*W = I[0~15]*W[0~15] + I[0~15]*W_constant[-8] (which is weight.min())
                output = output + outputF
        elif self.inference == 2:
            # original WAGE QCov2d
            _, inputQS, inputS = LinearQuantizeW(input, self.wl_input, input.max(), input.min())
            _, weightQS, weightS = LinearQuantizeW(self.weight, self.wl_weight, self.weight.max(), self.weight.min())
            #output = F.linear(input, self.weight, self.bias)
            output = F.linear(inputQS, weightQS, self.bias)
            _, output, _ = LinearQuantizeW(output, self.ADCprecision, output.max(), output.min())
        elif self.inference == 1:
            # original WAGE QCov2d
            _, inputQS, inputS = LinearQuantizeW(input, self.wl_input, input.max(), input.min())
            _, weightQS, weightS = LinearQuantizeW(self.weight, self.wl_weight, self.weight.max(), self.weight.min())
            #output = F.linear(input, self.weight, self.bias)
            output = F.linear(inputQS, weightQS, self.bias)
        else:
            output = outputOrignal
        return output
    
class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, 
                 wl_input=8,wl_weight=8,inference=1,subArray=128,ADCprecision=5,zero_point_opt=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.cellBit = 1                # SRAM
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.zero_point_opt = zero_point_opt
        
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
        s += ', ibits={wl_input}, wbits={wl_weight}, inference={inference}, subArray={subArray}, ADCbits={ADCprecision}'
        return s.format(**self.__dict__)
        
    def im2col(self, input):
        fold_param = dict(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_module = nn.Unfold(**fold_param)
        unfold_out = unfold_module(input)
        return unfold_out
    
    def forward(self, input):
        outputOrignal = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        
        # make input & weight to 2D tensors
        input2D = self.im2col(input).transpose(1,2)                                                            # [N, OH * OW, IC * KH * KW]
        weight2D = self.weight.reshape(self.weight.shape[0], -1)                                               # [IC * KH * KW, OC]

        self.linear_layer = QLinear(*weight2D.size(), self.bias, self.wl_input, self.wl_weight, self.inference, self.subArray, self.ADCprecision)
        self.linear_layer.weight.data = weight2D
        
        out = self.linear_layer(input2D).transpose(1,2).reshape(outputOrignal.size())
        return out