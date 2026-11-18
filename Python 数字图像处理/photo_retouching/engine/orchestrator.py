"""最小编排器：按模块声明的顺序逐个询问 plan()，执行需要做的，收集 Report。
后续在这里加：模块冲突检查（后做的模块是否改了前面保护的区域/频段）、修改预算、质检与局部回退。"""
from engine.module import Report


class Orchestrator:
    def __init__(self, modules):
        self.modules = sorted(modules, key=lambda m: m.order)

    def run(self, ctx):
        for m in self.modules:
            req, why = m.plan(ctx)
            if req is None:
                ctx.reports.append(Report(m.name, ran=False, skipped=why))
                continue
            rep = m.run(ctx, req)
            ctx.reports.append(rep)
        return ctx.image

    @staticmethod
    def summary(ctx, skipped=False):
        """执行了的模块及说明；skipped=True 时连同跳过的模块和原因（调试用）。"""
        parts = []
        for r in ctx.reports:
            if r.ran:
                parts.append(f"[{r.module}] {r.notes}")
            elif skipped:
                parts.append(f"[{r.module}] 跳过：{r.skipped}")
        return "；".join(parts) or "无需处理"
