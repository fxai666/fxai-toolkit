import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiTextPreview",
    beforeRegisterNodeDef: function (nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiTextPreview") return;

        var lastText = "";
        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // 自适应预览容器
            var previewBox = document.createElement("div");
            previewBox.style.width = "100%";
            previewBox.style.padding = "3px";
            previewBox.style.boxSizing = "border-box";
            previewBox.style.backgroundColor = "var(--comfy-input-bg)";
            previewBox.style.border = "1px solid var(--comfy-menu-border-color)";
            previewBox.style.borderRadius = "4px";
            previewBox.style.color = "var(--fg-color)";
            previewBox.style.whiteSpace = "pre-wrap";
            previewBox.style.wordBreak = "break-all";
            previewBox.style.overflowY = "auto";
            previewBox.style.overflowX = "hidden";
            previewBox.textContent = "";

            // 挂载DOM部件，高度自动计算
            this.addDOMWidget("text_preview", "custom", previewBox);
            this.previewBox = previewBox;

            // 初始化回填文本
            if (lastText) {
                previewBox.textContent = lastText;
            }

            return r;
        };

        // 执行后更新预览内容
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            if (message && message.text && this.previewBox) {
                lastText = message.text[0] || "";
                this.previewBox.textContent = lastText;
            }
        };

        // 配置加载/切换工作流恢复内容
        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (this.previewBox && lastText) {
                this.previewBox.textContent = lastText;
            }
            return r;
        };
    }
});