import { app } from "../../scripts/app.js";

// 全局请求锁：防止重复请求（解决疯狂点击卡顿）
let isRefreshing = false;

// 统一日志前缀 + 统一输出方法（只改这里，别处完全不动）
const LOG_PREFIX = "凤希AI友情提示：";
function comfyLog(msg, isError = false) {
    const text = LOG_PREFIX + msg;
    if (isError) {
        app.ui.dialog.showError(text);
    } else {
        app.ui.dialog.showStatus(text);
    }
}

    // 扩展主逻辑
    app.registerExtension({
        name: "FxAi.Prompt",
        async beforeRegisterNodeDef(t, nodeData) {
            if (nodeData.name === "FxAiPromptGenerator") {
                const orig = t.prototype.onNodeCreated;
                t.prototype.onNodeCreated = function () {
                    orig.apply(this, arguments);
                    const self = this;

                    // 刷新按钮
                    self.addWidget("button", "🔄 刷新模型", null, function () {
                        refreshModelsWithToast(self);
                    });
                    const systemWidget = self.widgets.find(w => w.name === "系统提示词");
                    if (systemWidget && !systemWidget.value) {
                        systemWidget.value = "";
                        self.setDirtyCanvas(true);
                    }

                    // 自动刷新
                    setTimeout(() => {
                        refreshModelsWithToast(self);
                    }, 1000);
                };
            }
}
});

function refreshModelsWithToast(node) {
    // 请求锁：正在加载时不重复执行（核心防卡）
    if (isRefreshing) {
        return;
    }

    const hostWidget = node.widgets.find(w => w.name === "API主机地址");
    
    if (!hostWidget || !hostWidget.value) {
        comfyLog("请先设置API主机地址");
        return;
    }
    
    // 开启加载锁
    isRefreshing = true;
    comfyLog("正在加载模型列表...");
    
    fetch(`/fxai/prompt/get_models?host=${encodeURIComponent(hostWidget.value)}`)
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(d => {
            isRefreshing = false; // 释放锁
            
            if (d.models && d.models.length > 0) {
                const sel = node.widgets.find(w => w.name === "模型选择");
                if (sel) {
                    sel.options.values = d.models;
                    sel.value = d.models[0];
                }
                node.setDirtyCanvas(true);
                comfyLog("模型列表加载成功");
            }
        })
        .catch(err => {
            isRefreshing = false; // 释放锁
            comfyLog("模型加载失败：" + err.message, true);
        });
}