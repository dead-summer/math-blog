#import "@preview/theorion:0.4.1": *
#import "@preview/cetz:0.3.4": canvas, draw


#let indicator = math.upright(math.bold("1"))
#let div = math.upright("div")
#let span = math.upright("span")
#let diag = math.upright("diag")
#let rank = math.upright("rank")
#let cond = math.upright("cond")

/// 绘制两层神经网络示意图
/// 参数：
///   d — 输入维度（输入层圆圈数）
///   n — 输出维度（输出层圆圈数）
///   y — 输出符号名称（字符串），如 "sigma"、"u"、"hat(y)"、"tilde(y)" 等
///   in-stroke — 输入层到隐藏层的连线样式
///   out-stroke — 隐藏层到输出层的连线样式
#let neural-net(
  d: 3,
  n: 2,
  y: "sigma",
  in-stroke: (paint: blue, thickness: 0.65pt),
  out-stroke: (paint: red, dash: "dashed", thickness: 0.65pt),
) = canvas({
  import draw: *

  // 布局参数
  let r     = 0.28   // 节点半径 (cm)
  let hgap  = 3.2    // 层间水平间距
  let vgap  = 1.05   // 节点纵向间距
  let loff  = 0.90   // 标签水平偏移

  // 第 i 个节点（共 total 个）的纵坐标：整体居中
  // 使用纯整数乘法，避免 float() 转换引发 cetz 坐标解析失败
  let vy(i, total) = (total - 1 - 2 * i) * vgap / 2

  // 三层横坐标
  let x0 = 0         // 输入层
  let x1 = hgap      // 隐藏层
  let x2 = 2 * hgap  // 输出层

  // 隐藏层：9 个槽，倒数第 2 为省略号
  let h-slots = 9
  let h-circs = (0, 1, 2, 3, 4, 5, 6, 8)

  // 连接线（先画，让节点压在线上方）
  // 输入层 → 隐藏层：蓝色实线
  for i in range(d) {
    for j in h-circs {
      line(
        (x0, vy(i, d)),
        (x1, vy(j, h-slots)),
        stroke: in-stroke
      )
    }
  }

  // 隐藏层 → 输出层：红色虚线
  for j in h-circs {
    for k in range(n) {
      line(
        (x1, vy(j, h-slots)),
        (x2, vy(k, n)),
        stroke: out-stroke
      )
    }
  }

  // 输入层：标签 → 箭头 → 圆圈
  for i in range(d) {
    let py = vy(i, d)
    let lx = x0 - loff
    // 标签 x_i
    content(
      (lx, py),
      eval("$x_(" + str(i + 1) + ")$", mode: "markup")
    )
    // 箭头
    line(
      (lx + 0.25, py), (x0 - r, py),
      mark: (end: ">", size: 0.18),
      stroke: 0.75pt + black
    )
    // 节点
    circle((x0, py), radius: r, fill: white, stroke: 1pt + black)
  }

  // 隐藏层：圆圈 + 竖向省略号
  for j in h-circs {
    circle((x1, vy(j, h-slots)), radius: r, fill: white, stroke: 1pt + black)
  }
  content((x1, vy(7, h-slots)), $dots.v$)

  // 输出层：圆圈 → 箭头 → 标签
  for k in range(n) {
    let oy = vy(k, n)
    let lx = x2 + loff
    // 节点
    circle((x2, oy), radius: r, fill: white, stroke: 1pt + black)
    // 箭头
    line(
      (x2 + r, oy), (lx - 0.25, oy),
      mark: (end: ">", size: 0.18),
      stroke: 0.75pt + black
    )
    // 标签 y_k
    content(
      (lx, oy),
      eval("$" + y + "_(" + str(k + 1) + ")$", mode: "markup")
    )
  }
})

#neural-net(d: 3, n: 6, y: "sigma")
