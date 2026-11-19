// ==============================================
// 凤希AI · 打开即加载指定工作流
// ==============================================
// 机制说明（2026-11-20 修订 v4）：
//   ComfyUI 的"默认工作流"本体 = 前端 bundle 里硬编码的 var K（已被 fxai_default_workflow.py 覆盖）
//   本扩展负责把"打开后真正加载的那个文件"锁死成 TARGETS[0]。
//   🔴 判定不能再只看标签/路径—— 路径对但画布内容是旧的（本地草稿/会话恢复）时，
//      必须比对画布节点指纹，认不出目标节点就强制重开。
//   ⚠️ 2026-11-20 用户要求：**不再自动展开左侧工作流侧边栏**，已移除该逻辑。
import { app } from "../../scripts/app.js";

const TARGETS = [
    "千问2.1图片处理",          // 生图
    "MiniMax影视剧场双采版",     // 影视剧场
];

// 每个目标工作流必须出现的特征节点（用于校验"画布内容真的换了"）
// 指录取自云端工作流文件真实节点类型（2026-11-20 核对）
const FINGERPRINT = {
    "千问2.1图片处理": ["FxAiQwenImage21Edit"],
    "MiniMax影视剧场双采版": ["FxAiMiniMaxImageToVideoV2", "FxAiScreenManager"],
};

const WINDOW_MS = 20000;
const POLL_MS = 400;

function log(...a) {
    try { console.log("[fxai-default]", ...a); } catch (e) {}
}

function fileOf(name) {
    return name.replace(/\.json$/i, "") + ".json";
}

function findWorkflow(store, name) {
    const list = store.persistedWorkflows || [];
    const file = fileOf(name);
    return list.find((w) => w.path === "workflows/" + file)
        || list.find((w) => w.fullFilename === file)
        || list.find((w) => String(w.fullFilename || "").startsWith(name + "."))
        || null;
}

function isTarget(store, wf) {
    if (!wf) return false;
    const path = String(wf.path || wf.fullFilename || "");
    return TARGETS.some((n) => {
        const file = fileOf(n);
        return path === "workflows/" + file || path.startsWith(n + ".");
    });
}

// 当前画布的节点类型集合
function canvasTypes() {
    try {
        return new Set((app.graph?._nodes || []).map((n) => n.type));
    } catch (e) {
        return new Set();
    }
}

// 🔴 画布内容校验：光看路径不够，必须看节点指纹是否对得上
function canvasMatches(targetName) {
    const need = FINGERPRINT[targetName] || [];
    if (!need.length) return false;
    const have = canvasTypes();
    return need.every((t) => have.has(t));
}

app.registerExtension({
    name: "fxai-default-workflow",
    async setup() {
        const started = Date.now();
        let finished = false;

        const tick = async () => {
            if (finished || Date.now() - started > WINDOW_MS) {
                finished = true;
                return;
            }
            const store = app.extensionManager?.workflow;
            if (!store) return;

            const open = store.openWorkflows || [];
            const cur = store.activeWorkflow;

            if (!cur || open.length === 0) return;      // 还在初始化
            const nodeCount = app.graph?._nodes?.length || 0;
            if (nodeCount === 0) return;                // 画布还没建好

            const targetName = TARGETS[0];

            // 🔴 路径对 + 画布指纹也对 → 才是真的在目标工作流上
            if (isTarget(store, cur) && canvasMatches(targetName)) {
                finished = true;
                log("目标工作流已在前台（路径+节点校验通过），跳过");
                return;
            }
            // 路径对但指纹不对 → 标签名对、画布内容错，必须重开
            if (isTarget(store, cur) && !canvasMatches(targetName)) {
                log("⚠️ 标签/路径是目标但画布节点不匹配（可能是旧草稿），强制重开");
            }

            let found = TARGETS.map((n) => findWorkflow(store, n)).filter(Boolean);
            if (found.length === 0) {
                try { await store.loadWorkflows(); } catch (e) {}
                found = TARGETS.map((n) => findWorkflow(store, n)).filter(Boolean);
            }
            if (found.length === 0) { finished = true; log("找不到目标工作流，跳过"); return; }

            finished = true;
            try {
                log("强制打开默认工作流 →", found.map((w) => w.path).join(", "));
                const prev = cur;
                await store.openWorkflow(found[0]);
                const rest = found.slice(1).map((w) => w.path);
                if (rest.length) {
                    store.openWorkflowsInBackground({ left: [], right: rest });
                }
                if (prev && prev !== store.activeWorkflow && prev.isTemporary) {
                    try { await store.closeWorkflow(prev); } catch (e) {}
                }
                if (location.search) {
                    history.replaceState(null, "", location.pathname);
                }
            } catch (e) {
                log("打开失败（不影响使用）：", e);
            }
        };

        const timer = setInterval(async () => {
            try { await tick(); } catch (e) { log(e); }
            if (finished) clearInterval(timer);
        }, POLL_MS);
    },
});
