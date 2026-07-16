import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiMultiLinePrompt",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiMultiLinePrompt") return;

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.lines = [];

            // 强制隐藏 lines_data 参数
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

            // 创建滚动容器
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

            // ========== 新增：批量输入按钮 ==========
            this.addWidget("button", "📋 批量输入", null, (function(node) {
                return function() {
                    openBatchPopup(node);
                };
            })(this));

            // 初始创建表头 + 第一行
            createHeader(this);

            // 只定义这一个宽度：既是默认宽度，也是最小宽度
            const FIXED_WIDTH = 820;

            // 1. 设置默认宽度
            this.size[0] = FIXED_WIDTH;
            this.setSize(this.computeSize());

            // 2. 强制限制最小宽度（不能拉窄）+ 刷新后保持宽度
            this.onResize = (size) => {
                if (size[0] < FIXED_WIDTH) {
                    this.size[0] = FIXED_WIDTH;
                    this.setSize([FIXED_WIDTH, size[1]]);
                }
            };

            return r;
        };

        // 加载配置（仅保留文本解析）
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
                if (Array.isArray(list)) {
                    while (this.lines.length > 0) {
                        removeLine(this, this.lines[0]);
                    }
                    for (var j = 0; j < list.length; j++) {
                        var item = list[j];
                        var text = "";
                        
                        if (Array.isArray(item)) {
                            text = item[0] || ""; // 仅保留文本字段
                        } else {
                            text = item || "";
                        }
                        addLine(this, text);
                    }
                }
            } catch (e) {
                console.error("FxAiMultiLinePrompt: 加载数据失败", e);
            }

            return r;
        };

        // 序列化保存（仅保留文本字段）
        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];

            if (this.linesDataWidget && this.lines) {
                var values = [];
                for (var i = 0; i < this.lines.length; i++) {
                    values.push(this.lines[i].value); // 仅保留文本
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

            return onSerialize ? onSerialize.apply(this, arguments) : o;
        };
    },
});

// ========== 新增：批量输入自定义弹窗（美观ComfyUI风格） ==========
function openBatchPopup(node) {
    // 背景遮罩
    const mask = document.createElement("div");
    mask.style.cssText = `
        position: fixed; top:0; left:0; width:100%; height:100%;
        background: rgba(0,0,0,0.6); z-index:9999;
        display: flex; align-items:center; justify-content:center;
    `;

    // 弹窗主体
    const dialog = document.createElement("div");
    dialog.style.cssText = `
        width: 700px; background: #2a2a2a; border: 1px solid #666;
        border-radius: 10px; padding: 15px; box-shadow: 0 0 20px #000;
        color: #fff; font-family: sans-serif;
    `;

    // 标题
    const title = document.createElement("div");
    title.textContent = "批量导入提示词";
    title.style.cssText = "font-size:16px; font-weight:bold; margin-bottom:8px; text-align:center;";

    // 说明
    const tip = document.createElement("div");
    tip.textContent = "输入 JSON 数组格式，例如：[\"提示词1\",\"提示词2\",\"提示词3\"]，导入后追加到现有列表末尾";
    tip.style.cssText = "font-size:12px; color:#aaa; margin-bottom:10px; line-height:1.4;";

    // 文本域
    const textarea = document.createElement("textarea");
    textarea.placeholder = "粘贴你的提示词数组...";
    textarea.style.cssText = `
        width: 100%; height: 320px; box-sizing: border-box;
        background: #1a1a1a; color: #fff; border: 1px solid #555;
        border-radius: 6px; padding: 10px; font-size:12px;
        font-family: monospace; resize: vertical;
    `;

    // 按钮栏
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

    // 关闭
    const close = () => document.body.removeChild(mask);
    cancelBtn.onclick = close;

    // 导入逻辑
    okBtn.onclick = () => {
        try {
            const v = textarea.value.trim();
            if (!v) return alert("请输入内容");
            const arr = JSON.parse(v);
            if (!Array.isArray(arr)) return alert("必须是数组格式");
            
            arr.forEach(t => addLine(node, String(t || "")));
            alert(`成功导入 ${arr.length} 条提示词`);
            close();
        } catch (e) {
            alert("格式错误：" + e.message);
        }
    };

    textarea.focus();
}

// 创建顶部表头标签（仅保留序号、文本、操作列）
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

    // 表头标签文本（移除时长、音频索引、转场列）
    var labels = [
        { text: "序号", width: "24px" },
        { text: "提示词文本", flex: 1 },
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

// 添加行（仅保留文本字段）
function addLine(node, defaultValue) {
    defaultValue = defaultValue || "";

    var idx = node.lines.length;
    var row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "flex-start";
    row.style.gap = "6px";
    row.style.width = "100%";
    row.style.marginBottom = "8px";
    row.style.boxSizing = "border-box";

    // 1. 行号
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

    // 2. 文本框（仅保留核心文本输入）
    var textarea = document.createElement("textarea");
    textarea.placeholder = "输入内容...";
    textarea.style.flex = "1";
    textarea.style.minWidth = "0";
    textarea.style.minHeight = "60px";
    textarea.style.padding = "6px 8px";
    textarea.style.borderRadius = "4px";
    textarea.style.fontFamily = "monospace";
    textarea.style.fontSize = "12px";
    textarea.style.border = "1px solid var(--comfy-menu-border-color)";
    textarea.style.backgroundColor = "var(--comfy-input-bg)";
    textarea.style.color = "var(--fg-color)";
    textarea.style.resize = "vertical";
    textarea.style.boxSizing = "border-box";
    textarea.value = defaultValue;

    // 3. 操作按钮（上移/下移/删除）
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

    // 组装行元素（移除时长、音频索引、转场相关DOM）
    row.appendChild(lineNumLabel);
    row.appendChild(textarea);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    // 行数据（仅保留文本相关字段）
    var item = {
        textarea: textarea,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        value: defaultValue,
        label: lineNumLabel
    };
    node.lines.push(item);

    // 文本变化监听
    textarea.addEventListener("input", function() {
        item.value = textarea.value;
        updateHidden(node);
    });

    // 上移按钮
    upBtn.onclick = function() {
        moveLine(node, item, -1);
    };

    // 下移按钮
    downBtn.onclick = function() {
        moveLine(node, item, 1);
    };

    // 删除按钮
    delBtn.onclick = function() {
        removeLine(node, item);
    };

    setTimeout(function() {
        node.scrollContainer.scrollTop = node.scrollContainer.scrollHeight;
    }, 10);

    updateHidden(node);
}

// 上移/下移行（逻辑不变）
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

    // 交换数组
    var temp = node.lines[index];
    node.lines[index] = node.lines[newIndex];
    node.lines[newIndex] = temp;

    // 交换DOM
    var container = node.scrollContainer;
    container.insertBefore(
        node.lines[newIndex].row,
        dir === -1 ? node.lines[index].row : node.lines[index].row.nextSibling
    );

    refreshLineNumbers(node);
    updateHidden(node);
}

// 刷新行号（逻辑不变）
function refreshLineNumbers(node) {
    for (var i = 0; i < node.lines.length; i++) {
        node.lines[i].label.textContent = (i + 1) + ".";
    }
}

// 删除行（逻辑不变）
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

// 更新隐藏数据（仅保留文本字段）
function updateHidden(node) {
    if (!node.linesDataWidget) return;

    var values = [];
    for (var i = 0; i < node.lines.length; i++) {
        values.push(node.lines[i].value);
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