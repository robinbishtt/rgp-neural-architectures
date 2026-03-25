import pytest
torch = pytest.importorskip("torch", reason="torch not installed")  
import torch
import torch.nn as nn
class TestDeviceManager:
    def test_returns_torch_device(self):
        from src.utils.device_manager import DeviceManager
        assert isinstance(DeviceManager().get_device(), torch.device)
    def test_to_device(self):
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager()
        y = dm.to_device(torch.randn(3, 3))
        assert isinstance(y, torch.Tensor)
    def test_model_to_device(self):
        from src.utils.device_manager import DeviceManager
        dm = DeviceManager()
        m = dm.model_to_device(nn.Linear(4, 4))
        assert isinstance(m, nn.Module)
    def test_device_info_dict(self):
        from src.utils.device_manager import DeviceManager
        info = DeviceManager().get_device_info()
        assert isinstance(info, dict)