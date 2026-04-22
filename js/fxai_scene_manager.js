import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiSceneManager",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiSceneManager") return;

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
            this.scrollContainer.style.minWidth = "500px";
            this.scrollContainer.style.margin = "5px 0";
            this.scrollContainer.style.paddingRight = "5px";
            this.scrollContainer.style.boxSizing = "border-box";

            this.addDOMWidget("lines_container", "container", this.scrollContainer);

            // ========== 修复1：创建表头 ==========
            createHeader(this);

            this.addWidget("button", "➕ 添加新场景", null, (function(node) {
                return function() {
                    addLine(node);
                };
            })(this));

            // ========== 修复2：默认创建一个空行 ==========
            if (this.lines.length === 0) {
                addLine(this);
            }

            const FIXED_WIDTH = 780;
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
                if (Array.isArray(list)) {
                    while (this.lines.length > 0) {
                        removeLine(this, this.lines[0]);
                    }
                    for (var j = 0; j < list.length; j++) {
                        var item = list[j];
                        var duration = 5;
                        var text = "";
                        var audioNo = 0;
                        var imgNo = 0;
                        var tailNeedle = -1;
                        var transition = 1;
                        
                        if (Array.isArray(item)) {
                            duration = Number(item[0]) || 5;
                            text = item[1] || "";
                            if(item.length >=3) audioNo = Number(item[2]) || 0;
                            if(item.length >=4) imgNo = Number(item[3]) || 0;
                            if(item.length >=5) tailNeedle = Number(item[4]) || -1;
                            if(item.length >=6) transition = Number(item[5]) || 1;
                        } else {
                            text = item || "";
                        }
                        addLine(this, text, duration, audioNo, imgNo, tailNeedle, transition);
                    }
                }
            } catch (e) {
                console.error("FxAiMultiLineText: 加载数据失败", e);
            }

            return r;
        };

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
                        this.lines[i].imgno,
                        this.lines[i].tailNeedle,
                        this.lines[i].transition,
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
        { text: "场景", width: "24px" },
        { text: "时长", width: "50px" },
        { text: "场景控制提示词", flex: 1 },
        { text: "音频索引", width: "55px" },
        { text: "图片索引", width: "55px" },
        { text: "尾帧位置", width: "65px" },
        { text: "转场", width: "60px" },
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

function addLine(node, defaultValue, defaultDuration, defaultAudioNo, defaultImgNo, defaultTailNeedle, defaultTransition) {
    defaultValue = defaultValue || "";
    defaultDuration = defaultDuration || 5;
    defaultAudioNo = defaultAudioNo || 0;
    defaultImgNo = defaultImgNo ?? -1;
    defaultTailNeedle = defaultTailNeedle ?? -1;
    defaultTransition = defaultTransition ?? 1;

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

    var imgnoInput = document.createElement("input");
    imgnoInput.type = "number";
    imgnoInput.min = "0";
    imgnoInput.step = "1";
    imgnoInput.placeholder = "编号";
    imgnoInput.style.width = "55px";
    imgnoInput.style.height = "28px";
    imgnoInput.style.padding = "0 6px";
    imgnoInput.style.borderRadius = "4px";
    imgnoInput.style.border = "1px solid var(--comfy-menu-border-color)";
    imgnoInput.style.backgroundColor = "var(--comfy-input-bg)";
    imgnoInput.style.color = "var(--fg-color)";
    imgnoInput.style.textAlign = "center";
    imgnoInput.style.flexShrink = "0";
    imgnoInput.style.marginTop = "2px";
    imgnoInput.value = defaultImgNo;

    // 尾帧位置 —— 纯数字输入框，和图片索引完全一样
    var tailNeedleInput = document.createElement("input");
    tailNeedleInput.type = "number";
    tailNeedleInput.step = "1";
    tailNeedleInput.placeholder = "尾针";
    tailNeedleInput.style.width = "65px";
    tailNeedleInput.style.height = "28px";
    tailNeedleInput.style.padding = "0 6px";
    tailNeedleInput.style.borderRadius = "4px";
    tailNeedleInput.style.border = "1px solid var(--comfy-menu-border-color)";
    tailNeedleInput.style.backgroundColor = "var(--comfy-input-bg)";
    tailNeedleInput.style.color = "var(--fg-color)";
    tailNeedleInput.style.textAlign = "center";
    tailNeedleInput.style.flexShrink = "0";
    tailNeedleInput.style.marginTop = "2px";
    tailNeedleInput.value = defaultTailNeedle;

    var transitionCheckbox = document.createElement("input");
    transitionCheckbox.type = "checkbox";
    transitionCheckbox.checked = defaultTransition === 1;
    transitionCheckbox.style.width = "20px";
    transitionCheckbox.style.height = "20px";
    transitionCheckbox.style.marginTop = "6px";
    transitionCheckbox.style.flexShrink = "0";
    transitionCheckbox.style.cursor = "pointer";

    var transitionLabel = document.createElement("span");
    transitionLabel.style.fontSize = "12px";
    transitionLabel.style.color = "var(--fg-color)";
    transitionLabel.style.marginLeft = "2px";
    transitionLabel.style.marginTop = "4px";
    transitionLabel.style.flexShrink = "0";

    var transitionContainer = document.createElement("div");
    transitionContainer.style.display = "flex";
    transitionContainer.style.alignItems = "center";
    transitionContainer.style.minWidth = "50px";
    transitionContainer.style.justifyContent = "center";
    transitionContainer.style.flexShrink = "0";
    transitionContainer.appendChild(transitionCheckbox);
    transitionContainer.appendChild(transitionLabel);

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
    row.appendChild(durationInput);
    row.appendChild(textarea);
    row.appendChild(audionoInput);
    row.appendChild(imgnoInput);
    row.appendChild(tailNeedleInput);
    row.appendChild(transitionContainer);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    var item = {
        textarea: textarea,
        durationInput: durationInput,
        audionoInput: audionoInput,
        imgnoInput: imgnoInput,
        tailNeedleInput: tailNeedleInput,
        transitionCheckbox: transitionCheckbox,
        transitionLabel: transitionLabel,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        value: defaultValue,
        duration: defaultDuration,
        audiono: defaultAudioNo,
        imgno: defaultImgNo,
        tailNeedle: defaultTailNeedle,
        transition: defaultTransition,
        label: lineNumLabel
    };
    node.lines.push(item);

    textarea.addEventListener("input", function() {
        item.value = textarea.value;
        updateHidden(node);
    });

    durationInput.addEventListener("input", function() {
        var val = parseFloat(durationInput.value) || 5;
        if (val < 0.1) val = 0.1;
        durationInput.value = val;
        item.duration = val;
        updateHidden(node);
    });
    
    audionoInput.addEventListener("input", function() {
        var val = parseInt(audionoInput.value) || 0;
        if (val < 0) val = 0;
        audionoInput.value = val;
        item.audiono = val;
        updateHidden(node);
    });

    imgnoInput.addEventListener("input", function() {
        var val = parseInt(imgnoInput.value) || -1;
        if (val < -1) val = -1;
        imgnoInput.value = val;
        item.imgno = val;
        updateHidden(node);
    });

    tailNeedleInput.addEventListener("input", function() {
        var val = parseInt(tailNeedleInput.value) || -1;
        tailNeedleInput.value = val;
        item.tailNeedle = val;
        updateHidden(node);
    });

    transitionCheckbox.addEventListener("change", function() {
        item.transition = transitionCheckbox.checked ? 1 : 0;
        updateHidden(node);
    });

    upBtn.onclick = function() {
        moveLine(node, item, -1);
    };

    downBtn.onclick = function() {
        moveLine(node, item, 1);
    };

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

function updateHidden(node) {
    if (!node.linesDataWidget) return;

    var values = [];
    for (var i = 0; i < node.lines.length; i++) {
        values.push([
            node.lines[i].duration,
            node.lines[i].value,
            node.lines[i].audiono,
            node.lines[i].imgno,
            node.lines[i].tailNeedle,
            node.lines[i].transition
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