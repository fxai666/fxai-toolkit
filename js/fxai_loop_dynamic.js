/*
 * 凤希AI · 循环节点动态槽位 + 中文连接点
 *
 * 后端把循环变量一次性声明成 initial_value1..19（while 是 0..19），这样
 * 任何接法都能通过校验；但全摆出来太占地方。这里在前端按「实际用了几个」
 * 决定显示几个，末尾永远留一个空槽，接上就自动长出下一个。
 *
 * 只动前端显示，不动 API 契约：槽位名仍是 initial_value1 / value1，
 * 所以老工作流一个字节都不用改。
 */
import { app } from "../../scripts/app.js";

const ANY = "*";

// offset: 第 0 个动态槽位的数字后缀（for 从 1 数，while 从 0 数）
// fixedOut: 非动态输出（flow / index），它们不参与裁剪
const CFG = {
    FxAiForLoopStart: { offset: 1, max: 19, fixedOut: ["flow", "index"], widgetLabels: { total: "循环次数" } },
    FxAiForLoopEnd: { offset: 1, max: 19, fixedOut: [], widgetLabels: {} },
    FxAiWhileLoopStart: { offset: 0, max: 20, fixedOut: ["flow"], widgetLabels: { condition: "条件" } },
    FxAiWhileLoopEnd: { offset: 0, max: 20, fixedOut: [], widgetLabels: { condition: "条件" } },
};

const IN_LABEL = { flow: "循环体", condition: "条件" };
const OUT_LABEL = { flow: "循环体", index: "序号" };

const isDynIn = (s) => s && typeof s.name === "string" && s.name.startsWith("initial_value");
const isDynOut = (s) => s && typeof s.name === "string" && s.name.startsWith("value");

function applyLabels(node, cfg) {
    let n = 0;
    for (const s of node.inputs || []) {
        if (isDynIn(s)) {
            s.label = "值" + ++n;
        } else if (IN_LABEL[s.name]) {
            s.label = IN_LABEL[s.name];
        }
    }
    n = 0;
    for (const s of node.outputs || []) {
        if (isDynOut(s)) {
            s.label = "值" + ++n;
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

    // 动态输入始终是尾部连续的一段，用位置当「第几个」最稳
    const dyn = node.inputs.filter(isDynIn);
    let lastUsed = -1;
    for (let i = 0; i < dyn.length; i++) {
        if (dyn[i].link != null) {
            lastUsed = i;
        }
    }

    // 用了 n 个就显示 n+1 个（末尾留一个空槽给下一次接）
    const want = Math.min(Math.max(lastUsed + 2, 1), cfg.max);

    // 1) 从尾部收掉多余的，只收没接线的
    while (dyn.length > want) {
        const idx = node.inputs.length - 1;
        const slot = node.inputs[idx];
        if (!isDynIn(slot) || slot.link != null) {
            break;
        }
        node.removeInput(idx);
        dyn.pop();
    }

    // 2) 不够就补，名字按 offset 递增
    while (dyn.length < want) {
        const slot = node.addInput("initial_value" + (cfg.offset + dyn.length), ANY);
        if (!slot) {
            break;
        }
        dyn.push(slot);
    }

    // 3) 输出跟着输入数量走
    const wantOut = cfg.fixedOut.length + want;
    while (node.outputs.length > wantOut) {
        const idx = node.outputs.length - 1;
        const slot = node.outputs[idx];
        if (!isDynOut(slot) || (slot.links && slot.links.length)) {
            break;
        }
        node.removeOutput(idx);
    }
    while (node.outputs.length < wantOut) {
        const ordinal = node.outputs.length - cfg.fixedOut.length;
        if (!node.addOutput("value" + (cfg.offset + ordinal), ANY)) {
            break;
        }
    }

    applyLabels(node, cfg);

    // 4) 槽位变了高度要跟上，宽度只增不减（别抢用户拉好的宽度）
    try {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    } catch (e) {
        // 节点还没进图，下次连接变化会再算
    }

    node.setDirtyCanvas?.(true, true);
}

// 工作流正在 configure 时链接还没恢复完，这时候裁槽位会把链接连根拔掉。
// 等到 configure 结束再算，最多等 120 帧。
function schedule(node, cfg) {
    if (node.__fxaiTimer) {
        clearTimeout(node.__fxaiTimer);
    }
    node.__fxaiTimer = setTimeout(() => {
        node.__fxaiTimer = null;
        if (node.__fxaiSync) {
            return;
        }
        if (app.configuringGraph) {
            // 加载途中只贴中文标签（不动槽位），槽位等 configure 收尾再算
            applyLabels(node, cfg);
            const wait = (node.__fxaiWait || 0) + 1;
            if (wait > 120) {
                return;
            }
            node.__fxaiWait = wait;
            schedule(node, cfg);
            return;
        }
        node.__fxaiWait = 0;
        node.__fxaiSync = true;
        try {
            sync(node, cfg);
        } finally {
            node.__fxaiSync = false;
        }
    }, 0);
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

        // onAfterGraphConfigured 是 ComfyUI 自己在 configure 收尾时回调的钩子
        // （RerouteNode 也用它），此刻链接已经恢复完毕
        const origAfter = node.onAfterGraphConfigured;
        node.onAfterGraphConfigured = function () {
            const ret = origAfter ? origAfter.apply(this, arguments) : undefined;
            schedule(this, cfg);
            return ret;
        };

        schedule(node, cfg);
    },
});
