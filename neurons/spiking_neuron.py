from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, base, layer
import math
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):

        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):

        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class BaseNode_adaspike(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):

        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):

        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, s: torch.Tensor):

        self.neuronal_charge(x, s)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class MpNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x)
        return self.v

class Ada_MpNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor, w: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x, w)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x, w)

        return self.v

class Ada_MpNode_adaspike(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        
        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)


        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x, w, s)
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x, w, s)
        return self.v

class Multi_Node(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        self.spike = self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor, last_mem: torch.Tensor):
        if last_mem is None:
            self.neuronal_charge(x)
            self.neuronal_fire()
            self.neuronal_reset()
        else:
            self.register_memory('v', last_mem)
            self.neuronal_charge(x)
            self.neuronal_fire()
            self.neuronal_reset()

        return self.spike, self.v

class MpLIFNode(MpNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class Mp_AdaLIFNode(Ada_MpNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.
        self.tau = tau
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor, w: torch.Tensor):
        tau = w.sigmoid()
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau

class Mp_AdaLIFNode_adaspike(Ada_MpNode_adaspike):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.
        self.tau = tau
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'
 
    def neuronal_charge(self, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor):
        tau = w.sigmoid()
        if self.v_reset is None:
            self.v = self.v + s * (x - self.v) * tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau

class MpIFNode(MpNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

class Mp_ParametricLIFNode(MpNode):
    def __init__(self, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()  #.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()

class Mp_ParametricLIFNode_modify(MpNode):
    def __init__(self, size_h, size_w, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.ones(size=[size_w, size_h])* init_w)
        #self.w = self.w * init_w   # test

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()  #.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()

class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class LIFNode_adaspike(BaseNode_adaspike):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor, s: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + s*(x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau


class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) * self.w.sigmoid()
        else:
            if self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()


