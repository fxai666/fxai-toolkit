import { app } from "../../scripts/app.js";

// 全局单例 Toast 控制（解决卡顿核心！）
let currentToast = null;
let toastTimeout = null;

// 工具函数：显示提示（修复版，绝不卡顿）
function showToast(message, type = "info", duration = 3000) {
    // 先销毁旧 Toast（永远只保留一个）
    if (currentToast && currentToast.parentNode) {
        currentToast.parentNode.removeChild(currentToast);
    }
    if (toastTimeout) {
        clearTimeout(toastTimeout);
        toastTimeout = null;
    }

    if (!document.getElementById('comfy-toast-styles')) {
        const style = document.createElement('style');
        style.id = 'comfy-toast-styles';
        style.textContent = `
            .comfy-toast {
                position: fixed;
                top: 10px;
                right: 20px;
                padding: 12px 20px;
                color: white;
                border-radius: 8px;
                z-index: 99999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 14px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                display: flex;
                align-items: center;
                gap: 8px;
                max-width: 400px;
                max-height:50px;
                word-break: break-word;
                animation: toast-slide-in 0.3s ease;
            }
            .comfy-toast-fade {
                animation: toast-fade-out 0.3s ease forwards;
            }
            .comfy-toast-info { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
            .comfy-toast-success { background: linear-gradient(135deg, #10b981, #047857); }
            .comfy-toast-error { background: linear-gradient(135deg, #ef4444, #b91c1c); }
            .comfy-toast-warning { background: linear-gradient(135deg, #f59e0b, #d97706); }
            
            @keyframes toast-slide-in {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes toast-fade-out {
                from { opacity: 1; }
                to { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
    
    const toast = document.createElement("div");
    toast.className = `comfy-toast comfy-toast-${type}`;
    currentToast = toast; // 保存单例
    
    let icon = "ℹ️";
    switch(type) {
        case "success": icon = "✅"; break;
        case "error": icon = "❌"; break;
        case "warning": icon = "⚠️"; break;
    }
    
    toast.innerHTML = `<span style="font-size: 16px;">${icon}</span><span>凤希AI友情提示：${message}</span>`;
    document.body.appendChild(toast);
    
    // 自动消失
    toastTimeout = setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.add("comfy-toast-fade");
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
                currentToast = null;
            }, 300);
        }
    }, duration);
    
    return toast;
}

    // 全局请求锁：防止重复请求（解决疯狂点击卡顿）
    let isRefreshing = false;

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
        showToast("请先设置API主机地址", "warning");
        return;
    }
    
    // 开启加载锁
    isRefreshing = true;
    const loadingToast = showToast("正在加载模型列表...", "info", 30000); // 最多等30秒
    
    fetch(`/fxai/prompt/get_models?host=${encodeURIComponent(hostWidget.value)}`)
        .then(r => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
        })
        .then(d => {
            isRefreshing = false; // 释放锁
            if (loadingToast.parentNode) loadingToast.parentNode.removeChild(loadingToast);
            
            if (d.models && d.models.length > 0) {
                const sel = node.widgets.find(w => w.name === "模型选择");
                if (sel) {
                    sel.options.values = d.models;
                    sel.value = d.models[0];
                }
                node.setDirtyCanvas(true);
                showToast("模型加载成功", "success");
            } else {
                showToast("未找到可用模型", "warning");
            }
        })
        .catch(err => {
            isRefreshing = false; // 释放锁
            if (loadingToast.parentNode) loadingToast.parentNode.removeChild(loadingToast);
            showToast(`加载失败：${err.message}`, "error");
        });
}