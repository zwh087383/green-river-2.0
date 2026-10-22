# 原生机械臂服务独立启动

`scripts/sdk_motion_host.py` 复用厂家的 `wheel_robot_control.xacro` 和 `nerv/arm_hw/ArmRokaeAr5` 插件，为左右臂分别生成模型描述。启动时使用厂家的 `urdf_publisher` 向各自控制器的 `robot_description` 主题发布，控制器和原生动作插件加载完成后关闭这两个临时发布进程；启动失败时也清理它们。因此动作服务不再依赖厂家整体启动程序中残留的模型发布进程。

不能将模型作为 controller_manager 的直接启动参数：该 SDK 在控制器构造完成后才注册控制器，过早加载硬件会得到 controller manager is not found。运动目标、关节顺序和硬件插件仍来自厂家配置，并验证生成模型仅包含对应一只手臂。

2026-10-21 验证：8 项相关检查通过；实机启动归位完成后执行归位并断电，左右臂断电反馈均为 1，上电按钮渲染条件通过；网页、语音和表情屏进程均保持原实例。挥手速度配置为抬手 0.5，挥动及收手 0.6。

验证记录：`data/runtime/motion-restart-verification-20261021-143611.json`。
