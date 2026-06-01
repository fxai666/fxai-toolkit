import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiTextPreview",
    beforeRegisterNodeDef: function (nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiTextPreview") {
            return;
        }

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onExecuted = nodeType.prototype.onExecuted;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // 找到隐藏的缓存 widget（和你多行文本一样）
            this.cacheWidget = null;
            for (var i = 0; i < this.widgets.length; i++) {
                var w = this.widgets[i];
                if (w && w.name === "cache_text") {
                    this.cacheWidget = w;
                    setTimeout(function(){
                        w.hidden = true;
                    },0);
                    break;
                }
            }

            // 创建预览框
            var previewBox = document.createElement("div");
            previewBox.style.width = "100%";
            previewBox.style.padding = "6px";
            previewBox.style.boxSizing = "border-box";
            previewBox.style.backgroundColor = "var(--comfy-input-bg)";
            previewBox.style.border = "1px solid var(--comfy-menu-border-color)";
            previewBox.style.borderRadius = "4px";
            previewBox.style.color = "var(--fg-color)";
            previewBox.style.whiteSpace = "pre-wrap";
            previewBox.style.wordBreak = "break-all";
            previewBox.style.overflowY = "auto";
            previewBox.style.overflowX = "hidden";
            previewBox.style.minHeight = "60px";

            this.addDOMWidget("text_preview", "custom", previewBox);
            this.previewBox = previewBox;

            return r;
        };

        // 执行后更新显示 + 存入隐藏 widget（核心）
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }

            if (!message || !message.text || !this.previewBox || !this.cacheWidget) {
                return;
            }

            var text = message.text[0] || "";
            this.previewBox.textContent = text;
            this.cacheWidget.value = text;
        };

        // 切换标签/加载工作流 反显（完全照抄你的写法）
        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (!o || !o.widgets_values || !this.previewBox) {
                return r;
            }

            var value = "";
            for (var i = 0; i < o.widgets_values.length; i++) {
                var w = o.widgets_values[i];
                if (w && w.name === "cache_data") {
                    value = w.value || "";
                    break;
                }
            }

            this.previewBox.textContent = value;
            return r;
        };

        // 保存到工作流（照抄你的）
        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];

            if (this.cacheWidget) {
                var val = this.cacheWidget.value || "";
                var found = false;
                for (var i = 0; i < o.widgets_values.length; i++) {
                    var w = o.widgets_values[i];
                    if (w && w.name === "cache_data") {
                        w.value = val;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    o.widgets_values.push({ name: "cache_data", value: val });
                }
            }

            return onSerialize ? onSerialize.apply(this, o) : o;
        };
    }
});