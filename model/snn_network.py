import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
#import model.MP_IFneuron
from neurons.spiking_neuron import *
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence

class BasicModel(nn.Module):
    '''
    Basic model class that can be saved and loaded
        with specified names.
    '''

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, path: str):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model')
        return self
    
    def to(self, device: torch.device):
        self.device = device
        super().to(device)
        return self

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(ConvLayer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out)
        return out

class Spike_recurrentConvLayer_nolstm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_recurrentConvLayer_nolstm, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, norm, tau, v_threshold, v_reset, activation_type)
        #self.recurrent_block = rnn.SpikingConvLSTMCell(input_size=out_channels, hidden_size=out_channels)

    def forward(self, x):
        x = self.conv(x)
        # state = self.recurrent_block(x, prev_state)
        # x = state[0]
        return x

class Spike_skip_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif'):
        super(Spike_skip_layer, self).__init__()
        self.conv = ConvLayer_ada_simmp(in_channels, out_channels, kernel_size, stride, padding, norm, tau, v_threshold, v_reset, activation_type)

    def forward(self, x, last_mem):
        x = self.conv(x, last_mem)
        return x

class ConvLayer_ada_simmp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif'):
        super(ConvLayer_ada_simmp, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        #self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.conv2d_pool = nn.Conv2d(out_channels, 1, kernel_size, stride, padding, bias=bias)
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        if activation_type == 'plif':
            self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'lif':
            self.activation = MpLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'if':
            self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'ada_lif':
            self.activation = Mp_AdaLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 4, 2*2)
        self.sigmoid = nn.Sigmoid()
        #A
        self.get_theta = get_theta_simmp(channels1 = in_channels, channels2 = out_channels, type='global', type1 = 'mix')
        #B
        #self.get_theta = get_theta(channels = in_channels, type='channel', type1 = 'fr')

    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)

        w = self.get_theta(x, out)
        out = self.activation(out, last_mem, w.unsqueeze(-1).unsqueeze(-1))
        return out

class get_theta_simmp(nn.Module):
    def __init__(self, channels1, channels2, reduction=4, type='global', type1 = 'max'):
        super(get_theta_simmp, self).__init__()
        self.channels = channels1
        self.fc1 = nn.Linear(channels1, channels1 // reduction)
        self.relu = nn.ReLU(inplace=True)
        if type == 'global':
            self.fc2 = nn.Linear(channels2// reduction, 1)
        elif type == 'channel':
            self.fc2 = nn.Linear(channels2 // reduction, channels1)
        self.sigmoid = nn.Sigmoid() 
        if type1 == 'max':    
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif type1 == 'fr':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif type1 == 'mix':
            self.pool = nn.AdaptiveMaxPool2d(1)
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            self.fc3 = nn.Linear(channels1+channels2, channels2 // reduction)

    def forward(self, x, x1):
        if x1 is None:
            theta = self.pool(x)
            theta = self.fc1(theta.squeeze(-1).squeeze(-1))
            theta = self.relu(theta)
            theta = self.fc2(theta)
        else:
            theta1 = self.pool(x1)
            theta2 = self.pool1(x)
            theta = torch.cat([theta1,theta2],1)
            theta = self.fc3(theta.squeeze(-1).squeeze(-1))
            theta = self.relu(theta)
            theta = self.fc2(theta)            
        return theta

class MP_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        super(MP_upsample_layer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        #self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        return out

class Spiking_residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, tau=2, v_threshold=1.0, v_reset=None):
        super(Spiking_residualBlock, self).__init__()
        bias = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.lif = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        #self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.lif(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.lif(out)
        return out

class Spike_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=None, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_upsample_layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan())        
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out = self.activation(out)

        return out

class TemporalFlatLayer_ada_simmp_concat(nn.Module):
    def __init__(self, tau=2.0, v_reset=None, activation_type='plif'):
        super(TemporalFlatLayer_ada_simmp_concat, self).__init__()

        self.conv2d = nn.Conv2d(64, 32, 1, bias=False)
        self.norm_layer = nn.BatchNorm2d(32)
        self.conv2d_pool = nn.Conv2d(32, 1, 1, bias=False)
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        if activation_type == 'plif':
            self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'lif':
            self.activation = MpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        elif activation_type == 'if':
            self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
        elif activation_type == 'ada_lif':
            self.activation = Mp_AdaLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())            
        self.get_theta = get_theta_simmp(channels1 = 64, channels2 = 32, type='global', type1 = 'mix')

    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        w = self.get_theta(x, out)
        out = self.activation(out, last_mem, w.unsqueeze(-1).unsqueeze(-1))
        return out

class TemporalFlatLayer_concat(nn.Module):
    def __init__(self, tau=2.0, v_reset=None):
        super(TemporalFlatLayer_concat, self).__init__()

        self.conv2d = nn.Conv2d(64, 1, 1, bias=False)
        self.norm_layer = nn.BatchNorm2d(1)
        self.activation = MpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan())
        #self.activation = MpIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan())
        #self.activation = Mp_ParametricLIFNode(v_threshold=float('Inf'), v_reset=v_reset, surrogate_function=surrogate.ATan()) 
                           
    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out, last_mem)
        return out

class PAEVSNN_LIF_AMPLIF_final(BasicModel):

    def __init__(self, kwargs = {}):
        super().__init__()
        activation_type =  kwargs['activation_type']
        mp_activation_type = kwargs['mp_activation_type']
        spike_connection = kwargs['spike_connection']
        v_threshold = kwargs['v_threshold']
        v_reset = kwargs['v_reset']
        tau = kwargs['tau']

        #header
        mp_activation_type = 'ada_lif'
        activation_type = 'lif'

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

        self.down1 = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down2 = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down3 = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.skip0 = Spike_skip_layer(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip1 = Spike_skip_layer(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip2 = Spike_skip_layer(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.skip3 = Spike_skip_layer(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)

        self.up1mp = Spike_skip_layer(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.up2mp = Spike_skip_layer(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)
        self.up3mp = Spike_skip_layer(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = mp_activation_type)

        self.aggregation1 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.aggregation2 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.aggregation3 = MP_upsample_layer(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, output_padding=0)

        self.residualBlock = nn.Sequential(
            Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )

        self.up1 = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up2 = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up3 = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.temporalflat = TemporalFlatLayer_ada_simmp_concat(tau = tau, v_reset=None, activation_type = mp_activation_type)

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 1, bias=False), 
        )
            
    def forward(self, on_img, prev_mem_states):
        if prev_mem_states is None:
            prev_mem_states = [None] * 8

        for i in range(on_img.size(1)):
            mem_states = []

            x = on_img[:,i,:,:].unsqueeze(dim=1)
            x_in = self.static_conv(x)  #head

            x1 = self.down1(x_in) #encoder
            x2 = self.down2(x1)
            x3 = self.down3(x2)

            s0 = self.skip0(x_in, prev_mem_states[0])
            mem_states.append(s0)
            s1 = self.skip1(x1, prev_mem_states[1])
            mem_states.append(s1)
            s2 = self.skip2(x2, prev_mem_states[2])
            mem_states.append(s2)
            s3 = self.skip3(x3, prev_mem_states[3])
            mem_states.append(s3)

            r1 = self.residualBlock(x3)

            u1 = self.up1(torch.cat([r1, x3],1)) #decoder
            u2 = self.up2(torch.cat([u1, x2],1))
            u3 = self.up3(torch.cat([u2, x1],1))

            up1mp = self.up1mp(r1, prev_mem_states[4])
            mem_states.append(up1mp)
            Mp1 = s3 + up1mp
            up2mp = self.up2mp(u1, prev_mem_states[5])
            mem_states.append(up2mp)
            Mp2 = s2 + up2mp
            up3mp = self.up3mp(u2, prev_mem_states[6])
            mem_states.append(up3mp)
            Mp3 = s1 + up3mp

            a1 = self.aggregation1(Mp1)
            a2 = self.aggregation2(a1+Mp2)
            a3 = self.aggregation3(a2+Mp3)

            membrane_potential = self.temporalflat(torch.cat([u3,x_in],1), prev_mem_states[7])
            mem_states.append(membrane_potential)
            
            membrane_potential = self.final(membrane_potential+a3+s0)

        return membrane_potential, mem_states

class EVSNN_LIF_final(BasicModel):

    def __init__(self, kwargs = {}):
        super().__init__()
        activation_type =  kwargs['activation_type']
        mp_activation_type = kwargs['mp_activation_type']
        spike_connection = kwargs['spike_connection']
        v_threshold = kwargs['v_threshold']
        v_reset = kwargs['v_reset']
        tau = kwargs['tau']

        #header
        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.down1 = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down2 = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.down3 = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.residualBlock = nn.Sequential(
            Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )

        self.up1 = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up2 = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)
        self.up3 = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=v_reset, activation_type = activation_type)

        self.temporalflat = TemporalFlatLayer_concat(tau = tau, v_reset=None)

            
    def forward(self, on_img, prev_mem_states):
        for i in range(on_img.size(1)):
            x = on_img[:,i,:,:].unsqueeze(dim=1)

            x_in = self.static_conv(x) #.unsqueeze(dim=1)

            x1 = self.down1(x_in)
            x2 = self.down2(x1)
            x3 = self.down3(x2)

            r1 = self.residualBlock(x3)

            u1 = self.up1(torch.cat([r1,x3],1))
            u2 = self.up2(torch.cat([u1, x2],1))
            u3 = self.up3(torch.cat([u2, x1],1))
            
            membrane_potential = self.temporalflat(torch.cat([u3,x_in],1), prev_mem_states)

        return membrane_potential