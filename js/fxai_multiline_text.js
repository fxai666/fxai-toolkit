import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "fxai.FxAiMultiLineText",

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiMultiLineText") return;

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
            this.scrollContainer.style.width = "100%";
            this.scrollContainer.style.margin = "5px 0";
            this.scrollContainer.style.paddingRight = "5px";
            this.scrollContainer.style.boxSizing = "border-box";

            this.addDOMWidget("lines_container", "container", this.scrollContainer, {
                serialize: false,
                hideOnZoom: false
            });

            this.addWidget("button", "➕ 添加行", null, (function(node) {
                return function() {
                    addLine(node);
                };
            })(this));

            // 初始创建表头 + 第一行
            createHeader(this);
            addLine(this);

            // 只定义这一个宽度：既是默认宽度，也是最小宽度
            const FIXED_WIDTH = 720;

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

            // 3. 右键刷新节点 → 强制恢复宽度（关键！）
            this.onAfterConfigure = () => {
                setTimeout(() => {
                    this.size[0] = FIXED_WIDTH;
                    this.setSize(this.computeSize());
                }, 10);
            };

            // ======================
            // 修复 分段循环复用 NaN
            // ======================
            for (const w of this.widgets) {
                if (w.name === "分段循环复用") {
                    setTimeout(function(){
                        w.value = isNaN(w.value) ? 1 : w.value;
                        w.onChange = () => { if (isNaN(w.value)) w.value = 1; };
                    },0)
                }
            }

            return r;
        };

        // 加载配置（新增转场字段解析）
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
                        var duration = 5;
                        var text = "";
                        var audioNo = 1;
                        var transition = 1; // 转场默认启用（1）
                        
                        if (Array.isArray(item)) {
                            duration = Number(item[0]) || 5;
                            text = item[1] || "";
                            if(item.length >=3) audioNo = Number(item[2]) || 1;
                            if(item.length >=4) transition = Number(item[3]) || 1; // 解析转场值
                        } else {
                            text = item || "";
                        }
                        addLine(this, text, duration, audioNo, transition);
                    }
                }
            } catch (e) {
                console.error("FxAiMultiLineText: 加载数据失败", e);
            }

            return r;
        };

        // 序列化保存（新增转场字段：秒数 → 文本 → 音频索引 → 转场）
        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];

            if (this.linesDataWidget && this.lines) {
                var values = [];
                for (var i = 0; i < this.lines.length; i++) {
                    values.push([
                        this.lines[i].duration,
                        this.lines[i].value,
                        this.lines[i].audiono,
                        this.lines[i].transition // 新增转场值
                    ]);
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

// 新增：创建顶部表头标签（增加转场列）
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

    // 表头标签文本（新增转场列）
    var labels = [
        { text: "序号", width: "24px" },
        { text: "时长", width: "50px" },
        { text: "提示词文本", flex: 1 },
        { text: "音频索引", width: "55px" },
        { text: "转场", width: "60px" }, // 新增转场表头
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

// 添加行：新增转场复选框参数（默认启用）
function addLine(node, defaultValue, defaultDuration, defaultAudioNo, defaultTransition) {
    defaultValue = defaultValue || "";
    defaultDuration = defaultDuration || 5;
    defaultAudioNo = defaultAudioNo || 1;
    defaultTransition = (defaultTransition === undefined || defaultTransition === null) ? 1 : defaultTransition; // 转场默认1（启用）

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

    // 2. 秒数输入框
    var durationInput = document.createElement("input");
    durationInput.type = "number";
    durationInput.min = "0.1";
    durationInput.step = "0.1";
    durationInput.placeholder = "秒";
    durationInput.style.width = "50px";
    durationInput.style.height = "28px";
    durationInput.style.padding = "0 6px";
    durationInput.style.borderRadius = "4px";
    durationInput.style.border = "1px solid var(--comfy-menu-border-color)";
    durationInput.style.backgroundColor = "var(--comfy-input-bg)";
    durationInput.style.color = "var(--fg-color)";
    durationInput.style.textAlign = "center";
    durationInput.style.flexShrink = "0";
    durationInput.style.marginTop = "2px";
    durationInput.value = defaultDuration;

    // 3. 文本框
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

    // 4. 音频索引输入框
    var audionoInput = document.createElement("input");
    audionoInput.type = "number";
    audionoInput.min = "0";
    audionoInput.step = "1";
    audionoInput.placeholder = "编号";
    audionoInput.style.width = "55px";
    audionoInput.style.height = "28px";
    audionoInput.style.padding = "0 6px";
    audionoInput.style.borderRadius = "4px";
    audionoInput.style.border = "1px solid var(--comfy-menu-border-color)";
    audionoInput.style.backgroundColor = "var(--comfy-input-bg)";
    audionoInput.style.color = "var(--fg-color)";
    audionoInput.style.textAlign = "center";
    audionoInput.style.flexShrink = "0";
    audionoInput.style.marginTop = "2px";
    audionoInput.value = defaultAudioNo;

    // 5. 转场复选框（新增）
    var transitionCheckbox = document.createElement("input");
    transitionCheckbox.type = "checkbox";
    transitionCheckbox.checked = defaultTransition === 1; // 1=启用（勾选），0=禁用（不勾选）
    transitionCheckbox.style.width = "20px";
    transitionCheckbox.style.height = "20px";
    transitionCheckbox.style.marginTop = "6px";
    transitionCheckbox.style.flexShrink = "0";
    // 复选框样式优化（可选）
    transitionCheckbox.style.cursor = "pointer";

    // 转场标签（可选，提升可读性）
    var transitionLabel = document.createElement("span");
    transitionLabel.style.fontSize = "12px";
    transitionLabel.style.color = "var(--fg-color)";
    transitionLabel.style.marginLeft = "2px";
    transitionLabel.style.marginTop = "4px";
    transitionLabel.style.flexShrink = "0";

    // 转场容器（包裹复选框+标签）
    var transitionContainer = document.createElement("div");
    transitionContainer.style.display = "flex";
    transitionContainer.style.alignItems = "center";
    transitionContainer.style.minWidth = "50px";
    transitionContainer.style.justifyContent = "center";
    transitionContainer.style.flexShrink = "0";
    transitionContainer.appendChild(transitionCheckbox);
    transitionContainer.appendChild(transitionLabel);

    // 6. 操作按钮（上移/下移/删除）
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

    // 组装行元素
    row.appendChild(lineNumLabel);
    row.appendChild(durationInput);
    row.appendChild(textarea);
    row.appendChild(audionoInput);
    row.appendChild(transitionContainer); // 新增转场列
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    // 行数据（新增transition字段）
    var item = {
        textarea: textarea,
        durationInput: durationInput,
        audionoInput: audionoInput,
        transitionCheckbox: transitionCheckbox,
        transitionLabel: transitionLabel,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        value: defaultValue,
        duration: defaultDuration,
        audiono: defaultAudioNo,
        transition: defaultTransition, // 转场值（1=启用，0=禁用）
        label: lineNumLabel
    };
    node.lines.push(item);

    // 文本变化监听
    textarea.addEventListener("input", function() {
        item.value = textarea.value;
        updateHidden(node);
    });

    // 秒数变化监听
    durationInput.addEventListener("input", function() {
        var val = parseFloat(durationInput.value) || 5;
        if (val < 0.1) val = 0.1;
        durationInput.value = val;
        item.duration = val;
        updateHidden(node);
    });
    
    // 音频索引变化监听
    audionoInput.addEventListener("input", function() {
        var val = parseInt(audionoInput.value) || 0;
        if (val < 0) val = 0; // 恢复成 0，不是 1
        audionoInput.value = val;
        item.audiono = val;
        updateHidden(node);
    });

    // 转场复选框变化监听（新增）
    transitionCheckbox.addEventListener("change", function() {
        item.transition = transitionCheckbox.checked ? 1 : 0;
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
        if (node.lines.length <= 1) {
            alert("至少保留一行文本");
            return;
        }
        removeLine(node, item);
    };

    setTimeout(function() {
        node.scrollContainer.scrollTop = node.scrollContainer.scrollHeight;
    }, 10);

    updateHidden(node);
}

// 上移/下移行（无修改，复用原有逻辑）
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

// 刷新行号（无修改）
function refreshLineNumbers(node) {
    for (var i = 0; i < node.lines.length; i++) {
        node.lines[i].label.textContent = (i + 1) + ".";
    }
}

// 删除行（无修改）
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

// 更新隐藏数据（新增转场字段）
function updateHidden(node) {
    if (!node.linesDataWidget) return;

    var values = [];
    for (var i = 0; i < node.lines.length; i++) {
        values.push([
            node.lines[i].duration,
            node.lines[i].value,
            node.lines[i].audiono,
            node.lines[i].transition // 新增转场值
        ]);
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