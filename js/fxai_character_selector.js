import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiCharacterSelector";

function getWidget(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name);
}

function addUI(node) {
    if (node.__fxaiCharacterSelectorUI) return;
    node.__fxaiCharacterSelectorUI = true;

    const container = document.createElement("div");
    container.style.cssText = `
        padding: 10px; display: flex; flex-direction: column; gap: 8px;
        border: 1px solid #555; border-radius: 6px; box-sizing: border-box;
        width: fit-content; min-width: 160px;
    `;

    const pickBtn = document.createElement("button");
    pickBtn.textContent = "🎭 选择角色";
    pickBtn.style.cssText = `
        width: 150px; padding: 8px 12px; border: none; border-radius: 4px;
        background: #4a8fff; color: #fff; cursor: pointer; font-size: 13px;
        box-sizing: border-box;
    `;
    container.appendChild(pickBtn);

    const preview = document.createElement("div");
    preview.style.cssText = "display:none;";
    container.appendChild(preview);

    const domWidget = typeof node.addDOMWidget === "function"
        ? node.addDOMWidget("character_selector", "character_selector", container, {
            serialize: false,
            hideOnZoom: false,
            getValue: () => "",
            setValue: () => {},
        })
        : null;
    if (domWidget) {
        domWidget.computeSize = () => [container.scrollWidth || 180, 230];
    }

    function resizeNode() {
        if (!domWidget) return;
        const size = domWidget.computeSize();
        node.size = [Math.max(container.scrollWidth || 180, size[0]), Math.max(230, size[1] + 40)];
        node.setSize?.(node.size);
        app.graph?.setDirtyCanvas(true, true);
    }

    function refreshInfo() {
        const widget = getWidget(node, "角色头像");
        const val = widget?.value || "";
        preview.innerHTML = "";
        if (!val) {
            preview.style.display = "none";
            return;
        }
        const parts = val.split("/");
        const sub = parts[0] || "";
        const fname = parts[1] || "";
        if (!sub || !fname) {
            preview.style.display = "none";
            return;
        }
        const img = document.createElement("img");
        img.src = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(sub)}&filename=${encodeURIComponent(fname)}&t=${Date.now()}`);
        img.style.cssText = "width:150px; height:150px; object-fit:cover; border-radius:6px; border:1px solid #444;";
        preview.appendChild(img);
        preview.style.display = "block";
    }

    pickBtn.onclick = function () {
        const widget = getWidget(node, "角色头像");
        const initStr = widget?.value || "";
        window.FxAiCharacterAssetsSelector(initStr).then(function (val) {
            if (val === undefined || val === null) return;
            const first = String(val).split(",")[0]?.trim() || "";
            if (widget) {
                widget.value = first;
                if (widget.callback) widget.callback(first);
            }
            refreshInfo();
            setTimeout(resizeNode, 50);
        });
    };

    const widget = getWidget(node, "角色头像");
    if (widget && !widget.__fxaiCharacterSelectorHooked) {
        const originalCallback = widget.callback;
        widget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            refreshInfo();
            setTimeout(resizeNode, 50);
            return result;
        };
        widget.__fxaiCharacterSelectorHooked = true;
    }

    requestAnimationFrame(refreshInfo);
    requestAnimationFrame(resizeNode);
}

app.registerExtension({
    name: "FxAiCharacterSelector",
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
