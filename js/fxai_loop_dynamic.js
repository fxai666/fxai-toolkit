/*
 * 凤希AI · 循环节点动态槽位 + 中文连接点
 *
 * 后端把循环变量一次性声明成 initial_value1..19（while 是 0..19），这样
 * 任何接法都能通过校验；但全摆出来太占地方。这里在前端按「实际用了几个」
 * 决定显示几个，末尾永远留一个空槽，接上就自动长出下一个。
 *
 * 只动前端显示，不动 API 契约：槽位名仍是 initial_value1 / value1，
 * 所以老工作流一个字节都不用改。
 *
 * 三个硬约束（都在真实浏览器里量过）：
 *   1. total / condition 这类「控件连接点」会被 ComfyUI 追加在动态槽位**后面**，
 *      所以动态槽位不一定在尾部，必须按真实索引删，不能假设「从尾部收」。
 *   2. 直接 splice(node.inputs) 会把连线弄丢；只有 removeInput / addInput
 *      会顺带修好 link 的 target_slot 索引。所以只能「按索引删 + 末尾追加」，不能重排。
 *   3. 打开工作流时节点级的 onAfterGraphConfigured **不会**被调用（只有粘贴、建子图才调），
 *      所以加载收尾必须挂扩展的 loadedGraphNode，另外用轮询兜底。
 */
import { app } from "../../scripts/app.js";

const ANY = "*";

// offset: 第 0 个动态槽位的数字后缀（for 从 1 数，while 从 0 数）
// fixedOut: 非动态输出（flow / index），它们不参与裁剪
// inLabel / outLabel: 左进右出各自的中文名
const CFG = {
    FxAiForLoopStart: {
        offset: 1, max: 19, fixedOut: ["flow", "index"],
        inLabel: "初始值", outLabel: "过程值",
        widgetLabels: { total: "循环次数", start_index: "开始索引" },
    },
    FxAiForLoopEnd: {
        offset: 1, max: 19, fixedOut: [],
        inLabel: "过程值", outLabel: "输出值",
        widgetLabels: {},
    },
    FxAiWhileLoopStart: {
        offset: 0, max: 20, fixedOut: ["flow"],
        inLabel: "初始值", outLabel: "过程值",
        widgetLabels: { condition: "条件" },
    },
    FxAiWhileLoopEnd: {
        offset: 0, max: 20, fixedOut: [],
        inLabel: "过程值", outLabel: "输出值",
        widgetLabels: { condition: "条件" },
    },
};

const IN_LABEL = { flow: "循环体", condition: "条件" };
const OUT_LABEL = { flow: "循环体", index: "序号" };

const isDynIn = (s) => s && typeof s.name === "string" && s.name.startsWith("initial_value");
const isDynOut = (s) => s && typeof s.name === "string" && s.name.startsWith("value");

// link / links 在新前端是废弃字段（控制台会告警），优先用官方判定
function inConnected(node, i) {
    if (typeof node.isInputConnected === "function") {
        return node.isInputConnected(i);
    }
    const s = node.inputs[i];
    return !!(s && (s.link != null || (s.linkIds && s.linkIds.length)));
}

function outConnected(node, i) {
    if (typeof node.isOutputConnected === "function") {
        return node.isOutputConnected(i);
    }
    const s = node.outputs[i];
    return !!(s && s.links && s.links.length);
}

// 下一个该用的数字后缀。中间可能有被保留下来的空洞，按最大值往上接最稳
function nextOrdinal(dyn, prefix, offset) {
    let max = -1;
    for (const s of dyn) {
        const n = parseInt(s.name.slice(prefix.length), 10);
        if (!isNaN(n) && n > max) {
            max = n;
        }
    }
    return max < 0 ? offset : max + 1;
}

function applyLabels(node, cfg) {
    let n = 0;
    for (const s of node.inputs || []) {
        if (isDynIn(s)) {
            s.label = cfg.inLabel + ++n;
        } else if (IN_LABEL[s.name]) {
            s.label = IN_LABEL[s.name];
        }
    }
    n = 0;
    for (const s of node.outputs || []) {
        if (isDynOut(s)) {
            s.label = cfg.outLabel + ++n;
        } else if (OUT_LABEL[s.name]) {
            s.label = OUT_LABEL[s.name];
        }
    }
    for (const w of node.widgets || []) {
        const text = cfg.widgetLabels[w.name];
        if (text) {
            w.label = text;
        }
    }
}

function sync(node, cfg) {
    if (!node.inputs || !node.outputs) {
        return;
    }

    // ---- 输入 ----
    let dyn = node.inputs.filter(isDynIn);
    let lastUsed = -1;
    for (let k = 0; k < dyn.length; k++) {
        if (inConnected(node, node.inputs.indexOf(dyn[k]))) {
            lastUsed = k;
        }
    }
    // 用了 n 个就显示 n+1 个（末尾留一个空槽给下一次接）
    const want = Math.min(Math.max(lastUsed + 2, 1), cfg.max);

    // 多的收掉：从后往前，只收没接线的
    for (let k = dyn.length - 1; k >= want; k--) {
        const idx = node.inputs.indexOf(dyn[k]);
        if (idx < 0 || inConnected(node, idx)) {
            continue;
        }
        node.removeInput(idx);
    }

    // 不够就补
    dyn = node.inputs.filter(isDynIn);
    while (dyn.length < want) {
        const name = "initial_value" + nextOrdinal(dyn, "initial_value", cfg.offset);
        const slot = node.addInput(name, ANY);
        if (!slot) {
            break;
        }
        dyn.push(slot);
    }

    // ---- 输出（跟着输入数量走）----
    const wantOut = cfg.fixedOut.length + want;

    let dynOut = node.outputs.filter(isDynOut);
    for (let k = dynOut.length - 1; k >= want; k--) {
        const idx = node.outputs.indexOf(dynOut[k]);
        if (idx < 0 || outConnected(node, idx)) {
            continue;
        }
        node.removeOutput(idx);
    }

    dynOut = node.outputs.filter(isDynOut);
    while (dynOut.length < wantOut - cfg.fixedOut.length) {
        const name = "value" + nextOrdinal(dynOut, "value", cfg.offset);
        const slot = node.addOutput(name, ANY);
        if (!slot) {
            break;
        }
        dynOut.push(slot);
    }

    applyLabels(node, cfg);

    // 槽位变了高度要跟上，宽度只增不减（别抢用户拉好的宽度）
    try {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    } catch (e) {
        // 节点还没进图，下次连接变化会再算
    }

    node.setDirtyCanvas?.(true, true);
}

// 工作流正在 configure 时链接还没恢复完，这时候裁槽位会把链接连根拔掉。
// 等到 configure 结束再算：加载这条路由 loadedGraphNode 收尾，这里再留一条轮询兜底。
function schedule(node, cfg) {
    if (node.__fxaiTimer) {
        return;
    }
    const tick = () => {
        node.__fxaiTimer = null;
        if (node.__fxaiSync) {
            return;
        }
        if (app.configuringGraph) {
            // 加载途中只贴中文标签（不动槽位）
            applyLabels(node, cfg);
            if ((node.__fxaiWait = (node.__fxaiWait || 0) + 1) < 900) {
                node.__fxaiTimer = setTimeout(tick, 16);
            }
            return;
        }
        node.__fxaiWait = 0;
        node.__fxaiSync = true;
        try {
            sync(node, cfg);
        } finally {
            node.__fxaiSync = false;
        }
    };
    node.__fxaiTimer = setTimeout(tick, 0);
}

app.registerExtension({
    name: "FxAi.LoopDynamicSlots",
    nodeCreated(node) {
        const cfg = CFG[node.comfyClass];
        if (!cfg) {
            return;
        }

        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            const ret = origConn ? origConn.apply(this, arguments) : undefined;
            if (!this.__fxaiSync) {
                schedule(this, cfg);
            }
            return ret;
        };

        // 粘贴、建子图时 ComfyUI 会显式调它；打开工作流那条路不会
        const origAfter = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            const ret = origAfter ? origAfter.apply(this, arguments) : undefined;
            schedule(this, cfg);
            return ret;
        };

        schedule(node, cfg);
    },

    // 打开工作流：graph.configure 跑完后按节点回调一次，此刻链接已经恢复完毕
    loadedGraphNode(node) {
        const cfg = CFG[node.comfyClass];
        if (cfg) {
            schedule(node, cfg);
        }
    },
});
