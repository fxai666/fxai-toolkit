import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiMultiPromptEditor",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiMultiPromptEditor") return;

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.lines = [];

            this.promptsDataWidget = null;
            for (var i = 0; i < this.widgets.length; i++) {
                var w = this.widgets[i];
                if (w && w.name === "prompts_data") {
                    this.promptsDataWidget = w;
                    setTimeout(() => { w.hidden = true; }, 0);
                    break;
                }
            }

            this.scrollContainer = document.createElement("div");
            this.scrollContainer.style.height = "100%";
            this.scrollContainer.style.overflowY = "auto";
            this.scrollContainer.style.overflowX = "hidden";
            this.scrollContainer.style.minWidth = "600px";
            this.scrollContainer.style.margin = "5px 0";
            this.scrollContainer.style.paddingRight = "5px";
            this.scrollContainer.style.boxSizing = "border-box";

            this.addDOMWidget("prompts_container", "container", this.scrollContainer);
            createHeader(this); // 创建自定义表头

            this.addWidget("button", "➕ 添加新的分段场景控制提示词", null, () => addLine(this));

            // 默认创建一行
            //if (this.lines.length === 0) addLine(this);

            const FIXED_WIDTH = 700;
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
                if (w && w.name === "prompts_data") {
                    data = w.value;
                    break;
                }
            }

            if (!data) return r;

            try {
                var list = JSON.parse(data);
                if (Array.isArray(list)) {
                    while (this.lines.length > 0) removeLine(this, this.lines[0]);
                    for (var j = 0; j < list.length; j++) {
                        addLine(this, item["提示词文本"] || "", Number(item["索引编号"]) || 0, Number(item["开始时间"]) || 0.0, Number(item["结束时间"]) || 0.0);
                    }
                }
            } catch (e) {
                console.error("FxAiMultiPromptEditor: 加载数据失败", e);
            }

            return r;
        };

        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];

            if (this.promptsDataWidget && this.lines) {
                // 序列化为对象数组格式（新增，支持JSON对象数组）
                var values = [];
                for (var i = 0; i < this.lines.length; i++) {
                    values.push({
                        "索引编号": this.lines[i].index,
                        "开始时间": this.lines[i].start,
                        "结束时间": this.lines[i].end,
                        "提示词文本": this.lines[i].value
                    });
                }
                var json = JSON.stringify(values);
                this.promptsDataWidget.value = json;

                var found = false;
                for (var i = 0; i < o.widgets_values.length; i++) {
                    var w = o.widgets_values[i];
                    if (w && w.name === "prompts_data") {
                        w.value = json;
                        found = true;
                        break;
                    }
                }
                if (!found) o.widgets_values.push({ name: "prompts_data", value: json });
            }
            return onSerialize ? onSerialize.apply(this, arguments) : o;
        };
    },
});

// 创建表头（移除权重列）
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
        { text: "序号", width: "30px" },
        { text: "索引编号", width: "60px" },
        { text: "开始时间", width: "80px" },
        { text: "结束时间", width: "80px" },
        { text: "提示词", flex: 1 },
        { text: "操作", width: "90px" } // 移除权重列
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

// 添加行（移除权重参数和输入框）
function addLine(node, defaultValue, defaultIndex, defaultStart, defaultEnd) {
    defaultValue = defaultValue || "";
    defaultIndex = defaultIndex ?? 0;
    defaultStart = defaultStart ?? 0.0;
    defaultEnd = defaultEnd ?? 0.0;

    var idx = node.lines.length;
    var row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "flex-start";
    row.style.gap = "6px";
    row.style.width = "100%";
    row.style.marginBottom = "8px";
    row.style.boxSizing = "border-box";

    // 1. 序号标签
    var lineNumLabel = document.createElement("span");
    lineNumLabel.textContent = (idx + 1) + ".";
    lineNumLabel.style.minWidth = "30px";
    lineNumLabel.style.textAlign = "center";
    lineNumLabel.style.color = "var(--fg-color)";
    lineNumLabel.style.opacity = "0.7";
    lineNumLabel.style.fontFamily = "monospace";
    lineNumLabel.style.fontSize = "12px";
    lineNumLabel.style.lineHeight = "1.5";
    lineNumLabel.style.marginTop = "6px";
    lineNumLabel.style.flexShrink = "0";

    // 2. 索引编号输入框
    var indexInput = document.createElement("input");
    indexInput.type = "number";
    indexInput.min = "0";
    indexInput.step = "1";
    indexInput.placeholder = "索引";
    indexInput.style.width = "60px";
    indexInput.style.height = "28px";
    indexInput.style.padding = "0 6px";
    indexInput.style.borderRadius = "4px";
    indexInput.style.border = "1px solid var(--comfy-menu-border-color)";
    indexInput.style.backgroundColor = "var(--comfy-input-bg)";
    indexInput.style.color = "var(--fg-color)";
    indexInput.style.textAlign = "center";
    indexInput.style.flexShrink = "0";
    indexInput.style.marginTop = "2px";
    indexInput.value = defaultIndex;

    // 3. 开始时间输入框
    var startInput = document.createElement("input");
    startInput.type = "number";
    startInput.min = "0";
    startInput.step = "0.1";
    startInput.placeholder = "开始";
    startInput.style.width = "80px";
    startInput.style.height = "28px";
    startInput.style.padding = "0 6px";
    startInput.style.borderRadius = "4px";
    startInput.style.border = "1px solid var(--comfy-menu-border-color)";
    startInput.style.backgroundColor = "var(--comfy-input-bg)";
    startInput.style.color = "var(--fg-color)";
    startInput.style.textAlign = "center";
    startInput.style.flexShrink = "0";
    startInput.style.marginTop = "2px";
    startInput.value = defaultStart;

    // 4. 结束时间输入框
    var endInput = document.createElement("input");
    endInput.type = "number";
    endInput.min = "0.1";
    endInput.step = "0.1";
    endInput.placeholder = "结束";
    endInput.style.width = "80px";
    endInput.style.height = "28px";
    endInput.style.padding = "0 6px";
    endInput.style.borderRadius = "4px";
    endInput.style.border = "1px solid var(--comfy-menu-border-color)";
    endInput.style.backgroundColor = "var(--comfy-input-bg)";
    endInput.style.color = "var(--fg-color)";
    endInput.style.textAlign = "center";
    endInput.style.flexShrink = "0";
    endInput.style.marginTop = "2px";
    endInput.value = defaultEnd;

    // 5. 提示词文本框
    var textarea = document.createElement("textarea");
    textarea.placeholder = "输入提示词...";
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

    // 6. 操作按钮（上移、下移、删除）
    var upBtn = document.createElement("button");
    upBtn.textContent = "↑";
    upBtn.title = "上移此行";
    upBtn.style.width = "25px";
    upBtn.style.height = "25px";
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
    downBtn.style.width = "25px";
    downBtn.style.height = "25px";
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
    delBtn.style.width = "25px";
    delBtn.style.height = "25px";
    delBtn.style.borderRadius = "4px";
    delBtn.style.border = "none";
    delBtn.style.cursor = "pointer";
    delBtn.style.fontWeight = "bold";
    delBtn.style.backgroundColor = "#c52222";
    delBtn.style.color = "#fff";
    delBtn.style.flexShrink = "0";
    delBtn.style.marginTop = "2px";

    // 组装行元素（移除权重输入框）
    row.appendChild(lineNumLabel);
    row.appendChild(indexInput);
    row.appendChild(startInput);
    row.appendChild(endInput);
    row.appendChild(textarea);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    // 行数据对象（移除weight相关）
    var item = {
        textarea: textarea,
        indexInput: indexInput,
        startInput: startInput,
        endInput: endInput,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        value: defaultValue,
        index: defaultIndex,
        start: defaultStart,
        end: defaultEnd,
        label: lineNumLabel
    };
    node.lines.push(item);

    // 绑定输入事件（移除权重相关）
    textarea.addEventListener("input", () => {
        item.value = textarea.value;
        updateHidden(node);
    });

    indexInput.addEventListener("input", () => {
        var val = parseInt(indexInput.value) || 0;
        if (val < 0) val = 0;
        indexInput.value = val;
        item.index = val;
        updateHidden(node);
    });

    startInput.addEventListener("input", () => {
        var val = parseFloat(startInput.value) || 0.0;
        if (val < 0) val = 0.0;
        startInput.value = val;
        item.start = val;
        updateHidden(node);
    });

    endInput.addEventListener("input", () => {
        var val = parseFloat(endInput.value) || 0.0;
        if (val < 0.1) val = 0.1;
        if (val < item.start) val = item.start + 0.1; // 结束时间不能小于开始时间
        endInput.value = val;
        item.end = val;
        updateHidden(node);
    });

    // 绑定按钮事件
    upBtn.onclick = () => moveLine(node, item, -1);
    downBtn.onclick = () => moveLine(node, item, 1);
    delBtn.onclick = () => {
        removeLine(node, item);
    };

    // 滚动到底部
    setTimeout(() => {
        node.scrollContainer.scrollTop = node.scrollContainer.scrollHeight;
    }, 10);

    updateHidden(node);
}

// 移动行（逻辑不变）
function moveLine(node, item, dir) {
    var index = node.lines.findIndex(l => l === item);
    if (index === -1) return;

    var newIndex = index + dir;
    if (newIndex < 0 || newIndex >= node.lines.length) return;

    // 交换数组元素
    [node.lines[index], node.lines[newIndex]] = [node.lines[newIndex], node.lines[index]];
    // 交换DOM元素
    node.scrollContainer.insertBefore(
        node.lines[newIndex].row,
        dir === -1 ? node.lines[index].row : node.lines[index].row.nextSibling
    );

    refreshLineNumbers(node);
    updateHidden(node);
}

// 刷新序号（逻辑不变）
function refreshLineNumbers(node) {
    node.lines.forEach((line, idx) => {
        line.label.textContent = (idx + 1) + ".";
    });
}

// 删除行（逻辑不变）
function removeLine(node, item) {
    item.row.remove();
    node.lines = node.lines.filter(l => l !== item);
    refreshLineNumbers(node);
    updateHidden(node);
}

// 更新隐藏的数据源（序列化改为对象数组）
function updateHidden(node) {
    if (!node.promptsDataWidget) return;

    // 改为对象数组格式（支持JSON对象解析）
    var values = node.lines.map(line => ({
        "索引编号": line.index,
        "开始时间": line.start,
        "结束时间": line.end,
        "提示词文本": line.value
    }));
    var data = JSON.stringify(values);
    node.promptsDataWidget.value = data;
    
    if (node.promptsDataWidget.inputEl) {
        node.promptsDataWidget.inputEl.value = data;
        const event = new Event("input", { bubbles: true, cancelable: true });
        node.promptsDataWidget.inputEl.dispatchEvent(event);
    }
}