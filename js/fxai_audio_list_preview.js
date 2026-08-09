import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiAudioListPreview";

function renderAudioList(node, message) {
    if (!node.__fxaiAudioListPreview) return;
    const container = node.__fxaiAudioListPreview.container;
    container.innerHTML = "";

    const items = Array.isArray(message?.audio) ? message.audio : [];
    if (!items.length) {
        container.innerHTML = '<span style="color:#999;font-size:12px;">暂无音频</span>';
        return;
    }

    items.forEach((item, index) => {
        const url = api.apiURL(`/view?filename=${encodeURIComponent(item.filename)}&subfolder=${encodeURIComponent(item.subfolder || "")}&type=${encodeURIComponent(item.type || "temp")}`);
        const wrap = document.createElement("div");
        wrap.style.cssText = `
            display:flex; flex-direction:column; gap:4px; padding:6px;
            background:#1a1a1a; border-radius:6px; border:1px solid #333;
        `;
        const label = document.createElement("div");
        label.textContent = `音频 ${index + 1}`;
        label.style.cssText = "color:#ccc; font-size:11px;";
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "metadata";
        audio.style.width = "100%";
        audio.src = url;
        wrap.append(label, audio);
        container.appendChild(wrap);
    });
}

app.registerExtension({
    name: "FxAiAudioListPreview",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== TARGET_CLASS) return;

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            renderAudioList(this, message);
            return result;
        };
    },
    async setup() {
        api.addEventListener("executed", (event) => {
            const { node_id, output } = event.detail;
            if (!output?.audio) return;
            const node = app.graph.getNodeById(node_id);
            if (node?.comfyClass === TARGET_CLASS) {
                renderAudioList(node, output);
            }
        });
    },
    async nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) return;

        const container = document.createElement("div");
        container.style.cssText = `
            display:flex; flex-direction:column; gap:6px;
            width:100%; max-height:260px; overflow-y:auto; padding:4px;
            box-sizing:border-box;
        `;
        container.innerHTML = '<span style="color:#999;font-size:12px;">等待渲染...</span>';

        const domWidget = typeof node.addDOMWidget === "function"
            ? node.addDOMWidget("audio_list_preview", "audio_list_preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue: () => "",
                setValue: () => {},
            })
            : null;

        node.__fxaiAudioListPreview = { container, domWidget };
        if (domWidget) {
            domWidget.computeSize = () => [320, 240];
        }
    },
});
