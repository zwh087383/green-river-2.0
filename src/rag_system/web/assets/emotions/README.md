# 小绿表情资源

这些资源是 96×48 的像素风眼部动画，供工控机显示页和后续 ROS2 DisplayNode 使用。

## 使用方式

读取同目录的 `manifest.json`，根据状态名选择帧序列：

- `idle`：中性待机眼睛
- `listening`：正在听，交替眨眼
- `thinking`：正在思考，旋转螺旋
- `speaking`：正在回答，眨眼加腮红
- `success`：回答完成，星星眼
- `error`：异常或失败，出汗眨眼
- `sleep`：长时间空闲
- `shy` / `smug`：可选的个性化状态

每个状态的帧文件都以 `{state}_{序号}.png` 命名。素材使用纯黑底，渲染时应保持 `image-rendering: pixelated`，不要做平滑插值。

显示页以 `manifest.json` 的 `layout.eye_anchors` 作为待机眼睛中心，并把全部状态映射到同一屏幕位置，以 `content_bounds` 计算统一缩放比例。少数素材的原始眼睛中心不同（如 `thinking`），在对应状态的 `eye_anchors` 中单独标注。腮红、汗滴等装饰不参与眼睛定位，但会被完整保留在画面内；增改帧素材时应同步更新 `content_bounds` 和必要的眼睛锚点。

`idle` 当前使用开心素材中的普通眼睛帧作为中性待机占位。后续如果补充独立的普通眨眼素材，只需要更新 `manifest.json`，不需要修改显示逻辑。
