"""2.0 自动摄影后期引擎：分析层（PhotoContext）+ 统一模块接口 + 编排器。

一张照片进来：PhotoContext 惰性地算一次人脸、皮肤、EXIF、场景等分析结果；编排器按各模块声明的
作用区域、通道和执行条件决定做哪些、按什么顺序；每个模块返回 Report 说明改了什么、跳过了什么。
"""
from engine.context import PhotoContext
from engine.module import Module, Report, Request
from engine.orchestrator import Orchestrator

__all__ = ["PhotoContext", "Module", "Report", "Request", "Orchestrator"]
