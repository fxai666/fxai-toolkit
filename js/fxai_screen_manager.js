import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FxAiScreenManager",
    beforeRegisterNodeDef: function(nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiScreenManager") return;

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : void 0;
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
            
            createHeader(this);

            var self = this;
            this.addWidget("button", "➕ 添加新场景", null, function() {
                addLine(self);
            });

            this.addWidget("button", "📋 批量输入", null, function() {
                openBatchPopup(self);
            });

            if (this.lines.length === 0) {
                addLine(this);
            }

            var FIXED_WIDTH = 840;
            this.size[0] = FIXED_WIDTH;
            this.setSize(this.computeSize());

            this.onResize = function(size) {
                if (size[0] < FIXED_WIDTH) {
                    this.size[0] = FIXED_WIDTH;
                    this.setSize([FIXED_WIDTH, size[1]]);
                }
            };

            return r;
        };

        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : void 0;
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
                        var duration = 10;
                        var text = "";
                        var imgStr = "";
                        
                        if (Array.isArray(item)) {
                            duration = Number(item[0]) || 10;
                            text = item[1] || "";
                            if (item.length >= 3) imgStr = String(item[2] || "");
                            // 兼容旧格式（图片在索引4）：适配旧数据
                            if (item.length >= 5 && !imgStr) imgStr = String(item[4] || "");
                        } else {
                            text = item || "";
                        }
                        addLine(this, text, duration, imgStr);
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
                        this.lines[i].imgStr,
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

function openBatchPopup(node) {
    var mask = document.createElement("div");
    mask.style.position = "fixed";
    mask.style.top = "0";
    mask.style.left = "0";
    mask.style.width = "100%";
    mask.style.height = "100%";
    mask.style.background = "rgba(0,0,0,0.6)";
    mask.style.zIndex = "9999";
    mask.style.display = "flex";
    mask.style.alignItems = "center";
    mask.style.justifyContent = "center";

    var dialog = document.createElement("div");
    dialog.style.width = "700px";
    dialog.style.background = "#2a2a2a";
    dialog.style.border = "1px solid #666";
    dialog.style.borderRadius = "10px";
    dialog.style.padding = "15px";
    dialog.style.boxShadow = "0 0 20px #000";
    dialog.style.color = "#fff";
    dialog.style.fontFamily = "sans-serif";

    var title = document.createElement("div");
    title.textContent = "批量导入场景提示词";
    title.style.fontSize = "16px";
    title.style.fontWeight = "bold";
    title.style.marginBottom = "8px";
    title.style.textAlign = "center";

var tip = document.createElement("div");
    tip.textContent = "输入 JSON 数组格式，例如：[\"场景提示词1\",\"场景提示词2\",\"场景提示词3\"]，导入后追加到现有列表末尾";
    tip.style.fontSize = "12px";
    tip.style.color = "#aaa";
    tip.style.marginBottom = "10px";
    tip.style.lineHeight = "1.4";

    var textarea = document.createElement("textarea");
    textarea.placeholder = "粘贴你的提示词数组...";
    textarea.style.width = "100%";
    textarea.style.height = "320px";
    textarea.style.boxSizing = "border-box";
    textarea.style.background = "#1a1a1a";
    textarea.style.color = "#fff";
    textarea.style.border = "1px solid #555";
    textarea.style.borderRadius = "6px";
    textarea.style.padding = "10px";
    textarea.style.fontSize = "12px";
    textarea.style.fontFamily = "monospace";
    textarea.style.resize = "vertical";

    var bar = document.createElement("div");
    bar.style.display = "flex";
    bar.style.justifyContent = "center";
    bar.style.gap = "10px";
    bar.style.marginTop = "12px";

    var okBtn = document.createElement("button");
    okBtn.textContent = "✅ 确认导入";
    okBtn.style.padding = "6px 14px";
    okBtn.style.background = "#4a86e8";
    okBtn.style.color = "#fff";
    okBtn.style.border = "none";
    okBtn.style.borderRadius = "4px";
    okBtn.style.cursor = "pointer";

    var cancelBtn = document.createElement("button");
    cancelBtn.textContent = "❌ 取消";
    cancelBtn.style.padding = "6px 14px";
    cancelBtn.style.background = "#666";
    cancelBtn.style.color = "#fff";
    cancelBtn.style.border = "none";
    cancelBtn.style.borderRadius = "4px";
    cancelBtn.style.cursor = "pointer";

    bar.appendChild(okBtn);
    bar.appendChild(cancelBtn);
    dialog.appendChild(title);
    dialog.appendChild(tip);
    dialog.appendChild(textarea);
    dialog.appendChild(bar);
    mask.appendChild(dialog);
    document.body.appendChild(mask);

    var close = function() {
        document.body.removeChild(mask);
    };
    cancelBtn.onclick = close;

    okBtn.onclick = function() {
        try {
            var v = textarea.value.trim();
            if (!v) {
                alert("请输入内容");
                return;
            }
            var arr = JSON.parse(v);
            if (!Array.isArray(arr)) {
                alert("必须是数组格式");
                return;
            }

            for (var k = 0; k < arr.length; k++) {
                addLine(node, String(arr[k] || ""));
            }
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
        { text: "场景", width: "30px" },
        { text: "时长(秒)", width: "60px" },
        { text: "素材", width: "480px" },
        { text: "台词", flex: 1 },
        { text: "操作", width: "90px" }
    ];

    for (var m = 0; m < labels.length; m++) {
        var item = labels[m];
        var span = document.createElement("span");
        span.textContent = item.text;
        span.style.textAlign = "center";
        if (item.width) span.style.minWidth = item.width;
        if (item.flex) span.style.flex = item.flex;
        span.style.flexShrink = "0";
        header.appendChild(span);
    }

    node.scrollContainer.appendChild(header);
}

function addLine(node, defaultValue, defaultDuration, defaultImgStr) {
    if (defaultValue === undefined) defaultValue = "";
    if (defaultDuration === undefined) defaultDuration = 10;
    if (defaultImgStr === undefined) defaultImgStr = "";

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
    lineNumLabel.style.minWidth = "30px";
    lineNumLabel.style.textAlign = "center";
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
    durationInput.style.width = "60px";
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

    // 素材缩略图区：flex 换行网格（每行 4 张图 + "＋"），点击打开图片多选选择器，每张可删除
    var materialBox = document.createElement("div");
    materialBox.style.width = "480px";
    materialBox.style.minHeight = "94px";
    materialBox.style.padding = "4px";
    materialBox.style.borderRadius = "4px";
    materialBox.style.border = "1px dashed var(--comfy-menu-border-color)";
    materialBox.style.backgroundColor = "var(--comfy-input-bg)";
    materialBox.style.boxSizing = "border-box";
    materialBox.style.cursor = "pointer";
    materialBox.style.display = "flex";
    materialBox.style.flexWrap = "wrap";
    materialBox.style.alignContent = "flex-start";
    materialBox.style.gap = "4px";
    materialBox.style.flexShrink = "0";

    materialBox.onclick = function (e) {
        if (e.target && e.target.tagName === "IMG") return;
        openMaterialSelector(item).then(function () {
            renderMaterialBox();
            updateHidden(node);
        });
    };

    function openMaterialSelector(item) {
        var initStr = (item.imgStr === undefined || item.imgStr === null) ? "" : String(item.imgStr);
        return FxAiCharacterAssetsSelector(initStr).then(function (val) {
            if (val !== undefined) {
                item.imgStr = String(val);
            }
        });
    }

    // 渲染两列缩略图 + 序号角标 + 删除角
    function renderMaterialBox() {
        materialBox.innerHTML = "";
        var imgs = (item.imgStr || "").split(",").map(function (s) { return s.trim(); }).filter(function (s) { return s !== ""; });
        if (imgs.length === 0) {
            var tip = document.createElement("span");
            tip.textContent = "＋ 选择素材";
            tip.style.cssText = "color:#999;font-size:12px;padding:30px 8px;display:block;text-align:center;";
            materialBox.appendChild(tip);
            return;
        }
        imgs.forEach(function (path, index) {
            var parts = path.split("/");
            var sub = parts[0] || "";
            var fname = parts[1] || "";
            var wrap = document.createElement("div");
            wrap.style.width = "90px";
            wrap.style.flexShrink = "0";
            wrap.style.position = "relative";

            var img = document.createElement("img");
            img.style.width = "90px";
            img.style.height = "90px";
            img.style.objectFit = "cover";
            img.style.display = "block";
            img.style.borderRadius = "4px";
            img.style.border = "1px solid var(--comfy-menu-border-color)";
            img.onerror = function () { img.style.opacity = "0.3"; };
            if (sub && fname) {
                img.src = api.apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(sub) + "&filename=" + encodeURIComponent(fname));
            }

            var numTag = document.createElement("div");
            numTag.textContent = index + 1;
            numTag.style.cssText = "position:absolute;top:0;left:0;width:18px;height:18px;background:rgba(0,0,0,0.6);color:#fff;font-size:11px;text-align:center;line-height:18px;border-radius:0 0 4px 0;z-index:2;";

            var delBtn = document.createElement("div");
            delBtn.textContent = "×";
            delBtn.style.cssText = "position:absolute;top:0;right:0;width:18px;height:18px;background:#f54242;color:#fff;text-align:center;line-height:18px;font-size:13px;cursor:pointer;z-index:3;border-radius:0 0 0 4px;";
            delBtn.onmousedown = function (e) { e.stopPropagation(); };
            delBtn.onclick = function (e) {
                e.stopPropagation();
                var arr = (item.imgStr || "").split(",").map(function (s) { return s.trim(); }).filter(function (s) { return s !== ""; });
                arr.splice(index, 1);
                item.imgStr = arr.join(",");
                renderMaterialBox();
                updateHidden(node);
            };

            wrap.appendChild(img);
            wrap.appendChild(numTag);
            wrap.appendChild(delBtn);

            var picLabel = document.createElement("input");
            picLabel.type = "text";
            picLabel.value = "<Picture " + (index + 1) + ">";
            picLabel.title = "双击选中便于复制（对应第 " + (index + 1) + " 张素材）";
            picLabel.style.cssText = "width:100%;height:16px;box-sizing:border-box;margin-top:3px;padding:0 2px;font-size:10px;text-align:center;color:var(--fg-color);background:var(--comfy-input-bg);border:1px solid var(--comfy-menu-border-color);border-radius:3px;";
            picLabel.onclick = function (e) { e.stopPropagation(); };
            picLabel.onmousedown = function (e) { e.stopPropagation(); };
            picLabel.onfocus = function () { picLabel.select(); };
            wrap.appendChild(picLabel);

            materialBox.appendChild(wrap);
        });
        // 追加"+"：未超过 8 张可继续添加（高度对齐图片高度）
        if (imgs.length < 8) {
            var plus = document.createElement("div");
            plus.textContent = "＋";
            plus.style.cssText = "width:90px;height:90px;border:1px dashed #555;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#999;font-size:22px;flex-shrink:0;background:transparent;margin-top:0;";
            materialBox.appendChild(plus);
        }
    }

    var textarea = document.createElement("textarea");
    textarea.placeholder = "输入台词...";
    textarea.style.flex = "1";
    textarea.style.minWidth = "0";
    textarea.style.minHeight = "109px";
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

    row.appendChild(lineNumLabel);
    row.appendChild(durationInput);
    row.appendChild(materialBox);
    row.appendChild(textarea);
    row.appendChild(upBtn);
    row.appendChild(downBtn);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    var item = {
        textarea: textarea,
        durationInput: durationInput,
        upBtn: upBtn,
        downBtn: downBtn,
        row: row,
        value: defaultValue,
        duration: defaultDuration,
        imgStr: defaultImgStr,
        label: lineNumLabel
    };
    node.lines.push(item);

    renderMaterialBox();

    textarea.addEventListener("input", function() {
        item.value = textarea.value;
        updateHidden(node);
    });

    durationInput.addEventListener("input", function() {
        var val = parseFloat(durationInput.value) || 10;
        if (val < 0.1) val = 0.1;
        durationInput.value = val;
        item.duration = val;
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

function updateHidden(node) {
    if (!node.linesDataWidget) return;

    var values = [];
    for (var i = 0; i < node.lines.length; i++) {
        values.push([
            node.lines[i].duration,
            node.lines[i].value,
            node.lines[i].imgStr
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