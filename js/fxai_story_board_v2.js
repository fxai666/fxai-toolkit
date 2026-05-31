import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxaiStoryBoardV2",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FxaiStoryBoardV2") return;

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.lines = [];

            this.linesDataWidget = null;
            for (var i = 0; i < this.widgets.length; i++) {
                var w = this.widgets[i];
                if (w && w.name === "lines_data") {
                    this.linesDataWidget = w;
                    setTimeout(function(){
                        w.hidden = true;
                    },0);
                    break;
                }
            }

            this.scrollContainer = document.createElement("div");
            this.scrollContainer.style.height = "100%";
            this.scrollContainer.style.overflowY = "auto";
            this.scrollContainer.style.overflowX = "hidden";
            this.scrollContainer.style.minWidth = "800px";
            this.scrollContainer.style.margin = "5px 0";
            this.scrollContainer.style.paddingRight = "5px";
            this.scrollContainer.style.boxSizing = "border-box";

            this.addDOMWidget("lines_container", "container", this.scrollContainer);

            this.addWidget("button", "➕ 添加行", null, (function(node) {
                return function() {
                    addLine(node);
                };
            })(this));

            this.addWidget("button", "📋 批量输入", null, (function(node) {
                return function() {
                    openBatchPopup(node);
                };
            })(this));

            createHeader(this);
            addLine(this);

            const FIXED_WIDTH = 950;
            this.size[0] = FIXED_WIDTH;
            this.setSize(this.computeSize());

            this.onResize = (size) => {
                if (size[0] < FIXED_WIDTH) {
                    this.size[0] = FIXED_WIDTH;
                    this.setSize([FIXED_WIDTH, size[1]]);
                }
            };

            return r;
        };

        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (!o || !o.widgets_values) return r;

            var data = null;
            for (var i = 0; i < o.widgets_values.length; i++) {
                var w = o.widgets_values[i];
                if (w && w.name === "lines_data") {
                    data = w.value;
                    break;
                }
            }

            if (!data) return r;

            try {
                var list = JSON.parse(data);
                while (this.lines.length > 0) {
                    removeLine(this, this.lines[0]);
                }
                for (var j = 0; j < list.length; j++) {
                    var item = list[j];
                    let promptText = item.提示词 || "";
                    let sceneProp = item.场景道具 || "";
                    addLine(this, promptText, sceneProp);
                }
            } catch (e) {
                console.error("FxaiStoryBoardV2: 加载数据失败", e);
            }

            return r;
        };

        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];

            if (this.linesDataWidget && this.lines) {
                var values = [];
                for (var i = 0; i < this.lines.length; i++) {
                    values.push({
                        提示词: this.lines[i].promptValue,
                        场景道具: this.lines[i].propValue
                    });
                }
                var json = JSON.stringify(values);
                this.linesDataWidget.value = json;

                var found = false;
                for (var i = 0; i < o.widgets_values.length; i++) {
                    var w = o.widgets_values[i];
                    if (w && w.name === "lines_data") {
                        w.value = json;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    o.widgets_values.push({ name: "lines_data", value: json });
                }
            }

            // 修复：onSerialize 不返回值，直接调用原方法并传递参数
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
        };
    },
});

function openBatchPopup(node) {
    const mask = document.createElement("div");
    mask.style.cssText = `
        position: fixed; top:0; left:0; width:100%; height:100%;
        background: rgba(0,0,0,0.6); z-index:9999;
        display: flex; align-items:center; justify-content:center;
    `;

    const dialog = document.createElement("div");
    dialog.style.cssText = `
        width: 700px; background: #2a2a2a; border: 1px solid #666;
        border-radius: 10px; padding: 15px; box-shadow: 0 0 20px #000;
        color: #fff; font-family: sans-serif;
    `;

    const title = document.createElement("div");
    title.textContent = "批量导入提示词";
    title.style.cssText = "font-size:16px; font-weight:bold; margin-bottom:8px; text-align:center;";

    const tip = document.createElement("div");
    tip.textContent = "只导入提示词字符串数组，例如：[\"提示词1\",\"提示词2\"]，场景道具会自动为空，可手动填写";
    tip.style.cssText = "font-size:12px; color:#aaa; margin-bottom:10px; line-height:1.4;";

    const textarea = document.createElement("textarea");
    textarea.placeholder = "粘贴字符串数组...";
    textarea.style.cssText = `
        width: 100%; height: 320px; box-sizing: border-box;
        background: #1a1a1a; color: #fff; border: 1px solid #555;
        border-radius: 6px; padding: 10px; font-size:12px;
        font-family: monospace; resize: vertical;
    `;

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex; justify-content:center; gap:10px; margin-top:12px;";

    const okBtn = document.createElement("button");
    okBtn.textContent = "✅ 确认导入";
    okBtn.style.cssText = "padding:6px 14px; background:#4a86e8; color:#fff; border:none; border-radius:4px; cursor:pointer;";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "❌ 取消";
    cancelBtn.style.cssText = "padding:6px 14px; background:#666; color:#fff; border:none; border-radius:4px; cursor:pointer;";

    bar.append(okBtn, cancelBtn);
    dialog.append(title, tip, textarea, bar);
    mask.append(dialog);
    document.body.append(mask);

    const close = () => document.body.removeChild(mask);
    cancelBtn.onclick = close;

    okBtn.onclick = () => {
        try {
            const v = textarea.value.trim();
            if (!v) return alert("请输入内容");
            const arr = JSON.parse(v);
            if (!Array.isArray(arr)) return alert("必须是字符串数组");
            
            arr.forEach(item => {
                let promptText = String(item || "");
                addLine(node, promptText, "");
            });
            alert(`成功导入 ${arr.length} 条提示词`);
            close();
        } catch (e) {
            alert("格式错误：" + e.message);
        }
    };

    textarea.focus();
}

function createHeader(node) {
    var header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.gap = "6px";
    header.style.width = "100%";
    header.style.marginBottom = "6px";
    header.style.paddingLeft = "2px";
    header.style.boxSizing = "border-box";
    header.style.fontSize = "12px";
    header.style.fontWeight = "bold";
    header.style.color = "#ffffff";

    var labels = [
        { text: "序号", width: "24px" },
        { text: "提示词文本", flex: 1 },
        { text: "场景道具", width: "120px" },
        { text: "操作", width: "90px" }
    ];

    labels.forEach(item => {
        var span = document.createElement("span");
        span.textContent = item.text;
        span.style.textAlign = "center";
        if (item.width) span.style.minWidth = item.width;
        if (item.flex) span.style.flex = item.flex;
        span.style.flexShrink = "0";
        header.appendChild(span);
    });

    node.scrollContainer.appendChild(header);
}

function addLine(node, promptDefault , propDefault ) {
    promptDefault = promptDefault || "";
    propDefault = propDefault || "";

    var idx = node.lines.length;
    var row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "flex-start";
    row.style.gap = "6px";
    row.style.width = "100%";
    row.style.marginBottom = "8px";
    row.style.boxSizing = "border-box";

    var lineNumLabel = document.createElement("span");
    lineNumLabel.textContent = (idx + 1) + ".";
    lineNumLabel.style.minWidth = "24px";
    lineNumLabel.style.textAlign = "right";
    lineNumLabel.style.color = "var(--fg-color)";
    lineNumLabel.style.opacity = "0.7";
    lineNumLabel.style.fontFamily = "monospace";
    lineNumLabel.style.fontSize = "12px";
    lineNumLabel.style.lineHeight = "1.5";
    lineNumLabel.style.marginTop = "6px";
    lineNumLabel.style.flexShrink = "0";

    var promptTextarea = document.createElement("textarea");
    promptTextarea.placeholder = "输入提示词...";
    promptTextarea.style.flex = "1";
    promptTextarea.style.minWidth = "0";
    promptTextarea.style.minHeight = "60px";
    promptTextarea.style.padding = "6px 8px";
    promptTextarea.style.borderRadius = "4px";
    promptTextarea.style.fontFamily = "monospace";
    promptTextarea.style.fontSize = "12px";
    promptTextarea.style.border = "1px solid var(--comfy-menu-border-color)";
    promptTextarea.style.backgroundColor = "var(--comfy-input-bg)";
    promptTextarea.style.color = "var(--fg-color)";
    promptTextarea.style.resize = "vertical";
    promptTextarea.style.boxSizing = "border-box";
    promptTextarea.value = promptDefault;

    var propInput = document.createElement("input");
    propInput.placeholder = "输入数字（如1.2）";
    propInput.style.minWidth = "120px";
    propInput.style.height = "30px";
    propInput.style.padding = "0 8px";
    propInput.style.borderRadius = "4px";
    propInput.style.fontFamily = "monospace";
    propInput.style.fontSize = "12px";
    propInput.style.border = "1px solid var(--comfy-menu-border-color)";
    propInput.style.backgroundColor = "var(--comfy-input-bg)";
    propInput.style.color = "var(--fg-color)";
    propInput.style.boxSizing = "border-box";
    propInput.style.marginTop = "6px";
    propInput.style.flexShrink = "0";
    propInput.value = propDefault;
    propInput.onclick = function(){
        FxAiCharacterAssetsSelector(this.value).then(result => {
            if(result)
            {
                this.value=result;
            }
        });
    }

    var upBtn = document.createElement("button");
    upBtn.textContent = "↑";
    upBtn.title = "上移此行";
    upBtn.style.width = "28px";
    upBtn.style.height = "28px";
    upBtn.style.borderRadius = "4px";
    upBtn.style.border = "none";
    upBtn.style.cursor = "pointer";
    upBtn.style.fontWeight = "bold";
    upBtn.style.backgroundColor = "#4a86e8";
    upBtn.style.color = "#fff";
    upBtn.style.flexShrink = "0";
    upBtn.style.marginTop = "2px";

    var downBtn = document.createElement("button");
    downBtn.textContent = "↓";
    downBtn.title = "下移此行";
    downBtn.style.width = "28px";
    downBtn.style.height = "28px";
    downBtn.style.borderRadius = "4px";
    downBtn.style.border = "none";
    downBtn.style.cursor = "pointer";
    downBtn.style.fontWeight = "bold";
    downBtn.style.backgroundColor = "#4a86e8";
    downBtn.style.color = "#fff";
    downBtn.style.flexShrink = "0";
    downBtn.style.marginTop = "2px";

    var delBtn = document.createElement("button");
    delBtn.textContent = "✕";
    delBtn.title = "删除此行";
    delBtn.style.width = "28px";
    delBtn.style.height = "28px";
    delBtn.style.borderRadius = "4px";
    delBtn.style.border = "none";
    delBtn.style.cursor = "pointer";
    delBtn.style.fontWeight = "bold";
    delBtn.style.backgroundColor = "#c52222";
    delBtn.style.color = "#fff";
    delBtn.style.flexShrink = "0";
    delBtn.style.marginTop = "2px";

    row.appendChild(lineNumLabel);
    row.appendChild(promptTextarea);
    row.appendChild(propInput);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    var item = {
        promptTextarea: promptTextarea,
        propInput: propInput,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        promptValue: promptDefault,
        propValue: propDefault,
        label: lineNumLabel
    };
    node.lines.push(item);

    promptTextarea.addEventListener("input", function() {
        item.promptValue = promptTextarea.value;
        updateHidden(node);
    });

    propInput.addEventListener("input", function() {
        item.propValue = propInput.value;
        updateHidden(node);
    });

    upBtn.onclick = function() {
        moveLine(node, item, -1);
    };

    downBtn.onclick = function() {
        moveLine(node, item, 1);
    };

    delBtn.onclick = function() {
        removeLine(node, item);
    };

    setTimeout(function() {
        node.scrollContainer.scrollTop = node.scrollContainer.scrollHeight;
    }, 10);

    updateHidden(node);
}

function moveLine(node, item, dir) {
    var index = -1;
    for (var i = 0; i < node.lines.length; i++) {
        if (node.lines[i] === item) {
            index = i;
            break;
        }
    }
    if (index === -1) return;

    var newIndex = index + dir;
    if (newIndex < 0 || newIndex >= node.lines.length) return;

    var temp = node.lines[index];
    node.lines[index] = node.lines[newIndex];
    node.lines[newIndex] = temp;

    var container = node.scrollContainer;
    container.insertBefore(
        node.lines[newIndex].row,
        dir === -1 ? node.lines[index].row : node.lines[index].row.nextSibling
    );

    refreshLineNumbers(node);
    updateHidden(node);
}

function refreshLineNumbers(node) {
    for (var i = 0; i < node.lines.length; i++) {
        node.lines[i].label.textContent = (i + 1) + ".";
    }
}

function removeLine(node, item) {
    item.row.remove();
    var newLines = [];
    for (var i = 0; i < node.lines.length; i++) {
        if (node.lines[i] !== item) {
            newLines.push(node.lines[i]);
        }
    }
    node.lines = newLines;
    refreshLineNumbers(node);
    updateHidden(node);
}

// 修复：updateHidden 函数移除 this，统一使用 node 参数
function updateHidden(node) {
    if (!node || !node.linesDataWidget) return;

    var values = [];
    for (var i = 0; i < node.lines.length; i++) {
        values.push({
            提示词: node.lines[i].promptValue,
            场景道具: node.lines[i].propValue
        });
    }
    var data = JSON.stringify(values);
    node.linesDataWidget.value = data;

    if (node.linesDataWidget.inputEl) {
        node.linesDataWidget.inputEl.value = data;
        var event = document.createEvent("Event");
        event.initEvent("input", true, true);
        node.linesDataWidget.inputEl.dispatchEvent(event);
    }
}