// ==============================================
// 凤希AI · 打开即加载指定的工作流（可多条）
// ==============================================
// 目的：全新打开 ComfyUI（裸地址、无历史）时，直接把下面 TARGETS 里列出的
// **已存在的持久化工作流**打开：标签名就是文件名、保存能回原文件、不新增文件夹。
//
// 为什么不建「默认」文件夹 / 不用 ?workflows= 参数：
//   前端 1.59 的 openFolderWorkflowsFromUrl() 只认单层文件夹名（isFolderName 禁止 "/"），
//   而且会把该文件夹里**全部**工作流都打开 —— 想让文件夹只开 2 条就得为它单独建目录。
//   直接用扩展列路径更精确，也不产生任何多余目录和标签。
//
// 行为边界（很保守，不抢用户操作）：
//   * 只在「当前只有一个临时图 + 无修改 + 本次会话还没切过」时动手
//   * 已经有真实工作流（上次恢复的）→ 完全不动
//   * 用户已经改动过画布 → 完全不动
//   * 页面打开 20 秒后不再插手
import { app } from "../../scripts/app.js";

// 想默认打开哪几条就写哪几条（不带 .json 也行，只写文件名主体）。
// 第一条放前台，其余在后台标签打开。
const TARGETS = [
    "千问2.1图片处理",          // 生图
    "MiniMax影视剧场双采版",     // 影视剧场
];

const WINDOW_MS = 20000;        // 只在启动这 20 秒内尝试
const POLL_MS = 400;

function log(...a) {
    try { console.log("[fxai-default]", ...a); } catch (e) {}
}

function findWorkflow(store, name) {
    const list = store.persistedWorkflows || [];
    const file = name.replace(/\.json$/i, "") + ".json";
    return list.find((w) => w.path === "workflows/" + file)
        || list.find((w) => w.fullFilename === file)
        || list.find((w) => String(w.fullFilename || "").startsWith(name + "."))
        || null;
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
            if (!cur.isTemporary) { finished = true; return; }   // 已是真实文件
            if (cur.isModified) { finished = true; return; }     // 用户改过
            if (open.length > 1) { finished = true; return; }    // 用户开了别的

            const nodeCount = app.graph?._nodes?.length || 0;
            if (nodeCount === 0) return;                // 画布还没建好

            let found = TARGETS.map((n) => findWorkflow(store, n)).filter(Boolean);
            if (found.length === 0) {
                try { await store.loadWorkflows(); } catch (e) {}
                found = TARGETS.map((n) => findWorkflow(store, n)).filter(Boolean);
            }
            if (found.length === 0) { finished = true; return; }

            finished = true;
            try {
                log("打开默认工作流 →", found.map((w) => w.path).join(", "));
                const temp = cur;
                await store.openWorkflow(found[0]);
                const rest = found.slice(1).map((w) => w.path);
                if (rest.length) {
                    store.openWorkflowsInBackground({ left: [], right: rest });
                }
                if (temp && temp !== store.activeWorkflow) {
                    await store.closeWorkflow(temp);
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
