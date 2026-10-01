import { app } from "../../scripts/app.js";

const TARGET_CLASS = "FxAiAudioSelector";

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function addUI(node) {
    if (node.__fxaiAudioSelectorUI) return;
    node.__fxaiAudioSelectorUI = true;

    const container = document.createElement("div");
    container.style.cssText = `
        padding: 8px; display: flex; flex-direction: column; gap: 6px;
        border: 1px solid #555; border-radius: 6px; box-sizing: border-box;
        width: 240px;
    `;

    const pickBtn = document.createElement("button");
    pickBtn.textContent = "📁 选择音频";
    pickBtn.style.cssText = `
        width: 100%; padding: 8px 12px; border: none; border-radius: 4px;
        background: #4a8fff; color: #fff; cursor: pointer; font-size: 13px;
        box-sizing: border-box;
    `;
    container.appendChild(pickBtn);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("audio_selector_ui", "audio_selector_ui", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;

    function resizeNode() {
        if (!domWidget) return;
        const width = 240;
        const height = 90;
        node.size = [Math.max(width, node.size?.[0] || 0), Math.max(height, (node.size?.[1] || 0))];
        app.graph?.setDirtyCanvas(true, true);
    }

    pickBtn.onclick = function () {
        if (typeof window.FxAiAudioSelector !== "function") {
            alert("音频选择弹窗未加载");
            return;
        }
        const widget = getWidget(node, "音频文件");
        const initPath = widget?.value || "";
        window.FxAiAudioSelector(initPath).then(function (result) {
            if (result === undefined) return;
            if (widget) {
                widget.value = result;
                if (widget.callback) widget.callback(result);
            }
            setTimeout(resizeNode, 50);
        });
    };

    requestAnimationFrame(resizeNode);
}

app.registerExtension({
    name: "FxAiAudioSelectorNode",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) return;
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            addUI(this);
            return result;
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) return;
        setTimeout(() => addUI(node), 200);
    },
});
