"""统一模块接口。

模块声明自己改什么（作用区域、通道、频段），编排器据此排顺序、检查冲突；
plan() 根据分析结果决定这张照片要不要做（返回 None 并说明原因即跳过），run() 执行并返回 Report。
编排器不定底层参数：只给 Request（作用区域、目标、要保护的东西、修改预算），具体半径、阈值由模块自己定。
"""
from dataclasses import dataclass, field


@dataclass
class Request:
    goal: str                                  # 例："reduce_skin_defects"、"technical_cleanup"
    region: object = None                      # None = 整图；否则 0~1 蒙版或 (x0, y0, x1, y1)
    preserve: list = field(default_factory=list)   # 例：["makeup", "texture", "face_structure"]
    budget: float = 1.0                        # 抽象的修改预算（0~1），不直接对应某个强度参数


@dataclass
class Report:
    module: str
    ran: bool
    notes: str = ""
    changed: object = None                     # 改动区域（蒙版或框），未知时 None
    channels: tuple = ()
    bands: tuple = ()
    skipped: str = ""                          # 没做或部分没做的原因
    confidence: float = 1.0


class Module:
    name = "module"
    region = "global"          # 声明的作用区域："person"、"skin"、"sky"、"global" ...
    channels = ("L", "ab")     # 改哪些通道
    bands = ("low", "mid", "high")
    geometry = False           # 是否改几何（拉直、裁剪、液化）
    order = 50                 # 默认执行顺序（小的先做）：几何 10、曝光 20、修复 30、人像 40、色彩 60、锐化 80

    def plan(self, ctx):
        """返回 (Request 或 None, 原因)。"""
        return Request(goal=self.name), ""

    def run(self, ctx, req):
        raise NotImplementedError
