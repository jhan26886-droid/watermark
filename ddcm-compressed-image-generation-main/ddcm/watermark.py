import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
# >>>ADD
import hashlib


class watermark_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number,
                seed: int = 123456, device: torch.device | None = None,
                dtype: torch.dtype = torch.float16):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.fpr = fpr
        self.user_number = user_number

        # >>>ADD
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.np_rs = np.random.RandomState(seed)           # 给 scipy/numpy 用
        self.torch_g = torch.Generator(device=self.device).manual_seed(self.seed + 1) # 给 torch 用

        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    # >>>ADD
    def _derive_bytes(self, tag: str, nbytes: int) -> bytes:
        # 从 (tag, seed) 派生固定字节串；避免依赖 get_random_bytes
        h = hashlib.sha256(f"{tag}:{self.seed}".encode("utf-8")).digest()
        out = b""
        ctr = 0
        while len(out) < nbytes:
            out += hashlib.sha256(h + ctr.to_bytes(4, "big")).digest()
            ctr += 1
        return out[:nbytes]

    ### >>> MODIFY (固定 key/nonce；删除 get_random_bytes)
    def stream_key_encrypt(self, sd):
        self.key = self._derive_bytes("wm_key", 32)
        self.nonce = self._derive_bytes("wm_nonce", 12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit
    
    # >>> MODIFY (绑定 random_state，返回到指定 device/dtype)
    def truncSampling(self, message):
        z = np.zeros(self.latentlength, dtype = np.float32)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            # 绑定 self.np_rs ，确保可复现
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1], random_state = self.np_rs)
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).to(device = self.device, dtype = self.dtype)
        return z

    ### >>> MODIFY (用内部 RNG 生成 watermark，比特按设备放置)
    def create_watermark_and_return_w(self):
        # watermark 比特：形状 [1, 4//ch, 64//hw, 64//hw]
        wm_np = self.np_rs.randint(0, 2, size = [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).astype(np.uint8)
        self.watermark = torch.from_numpy(wm_np).to(self.device)

        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy()) # m 是 {0,1} 比特
        w = self.truncSampling(m)     # (1,4,64,64)
        return w

    # >>> MODIFY (确保返回在当前设备)
    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8).to(self.device)
        return sd_tensor

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

    ### >>> ADD 到 class Gaussian_Shading_chacha 内部
    def export_meta(self) -> dict:
        """
        导出用于“事后验证”的元数据（与解压无关）：
        - kind/ch/hw/fpr/user_number/seed
        - watermark 比特（uint8 张量）
        - chacha 的 key/nonce（既保存 bytes 也保存 hex，便于人读）
        """
        meta = {
            "kind": "chacha",
            "ch": int(self.ch),
            "hw": int(self.hw),
            "fpr": float(self.fpr),
            "user_number": int(self.user_number),
            "seed": int(self.seed),
        }
        # watermark 比特（小：ch=1, hw=8 时为 256 bit）
        if self.watermark is not None:
            meta["watermark_bits"] = self.watermark.detach().cpu().to(torch.uint8)
        # chacha 的 key/nonce
        if self.key is not None:
            meta["key_bytes"] = self.key  # bytes
            meta["key_hex"] = self.key.hex()
        if self.nonce is not None:
            meta["nonce_bytes"] = self.nonce  # bytes
            meta["nonce_hex"] = self.nonce.hex()
        return meta






class watermark_xor:
    def __init__(self, ch_factor, hw_factor, fpr, user_number,
                seed: int = 123456, device: torch.device | None = None,
                dtype: torch.dtype = torch.float16):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.fpr = fpr
        self.user_number = user_number

        # >>>ADD
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.np_rs = np.random.RandomState(self.seed)
        self.torch_g = torch.Generator(device=self.device).manual_seed(self.seed + 1)

        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength
    
    # >>> MODIFY (绑定 random_state + 设备/精度)
    def truncSampling(self, message):
        z = np.zeros(self.latentlength, dtype=np.float32)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1], random_state=self.np_rs)
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).to(device=self.device, dtype=self.dtype)
        return z

    ### >>> MODIFY (用内部 RNG 生成 key/watermark)
    def create_watermark_and_return_w(self):
        key_np = self.np_rs.randint(0, 2, size=(1, 4, 64, 64)).astype(np.uint8)
        self.key = torch.from_numpy(key_np).to(self.device)
        wm_np = self.np_rs.randint(0, 2, size=(1, 4 // self.ch, 64 // self.hw, 64 // self.hw)).astype(np.uint8)
        self.watermark = torch.from_numpy(wm_np).to(self.device)

        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

    ### >>> ADD 到 class Gaussian_Shading 内部
    def export_meta(self) -> dict:
        """
        导出用于“事后验证”的元数据：
        - kind/ch/hw/fpr/user_number/seed
        - watermark 比特（uint8 张量）
        - xor 的 key（1x4x64x64 的 uint8 张量）
        """
        meta = {
            "kind": "xor",
            "ch": int(self.ch),
            "hw": int(self.hw),
            "fpr": float(self.fpr),
            "user_number": int(self.user_number),
            "seed": int(self.seed),
        }
        if self.watermark is not None:
            meta["watermark_bits"] = self.watermark.detach().cpu().to(torch.uint8)
        if self.key is not None:
            meta["key_bits"] = self.key.detach().cpu().to(torch.uint8)
        return meta




