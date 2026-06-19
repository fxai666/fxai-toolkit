import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
app.registerExtension({
    name: "FxAiLoraLoader",
    beforeRegisterNodeDef: function (nodeType, nodeData, app) {
        if (nodeData.name !== "FxAiLoraLoader") return;
        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.lines = [];
            this.loraDataWidget = null;
            for (var i = 0; i < this.widgets.length; i++) {
                var w = this.widgets[i];
                if (w && w.name === "lora_data") {
                    this.loraDataWidget = w;
                    w.hidden = true;
                    break;
                }
            }
            // 外层容器
            this.wrap = document.createElement("div");
            this.wrap.style.position = "relative";
            this.wrap.style.height = "100%";
            this.wrap.style.paddingBottom = "30px";
            this.wrap.style.boxSizing = "border-box";
            // 滚动容器
            this.scrollBox = document.createElement("div");
            this.scrollBox.style.height = "100%";
            this.scrollBox.style.minHeight = "100px";
            this.scrollBox.style.overflowY = "auto";
            this.scrollBox.style.overflowX = "hidden";
            this.scrollBox.style.paddingRight = "5px";
            this.wrap.appendChild(this.scrollBox);
            // 表格
            this.table = document.createElement("table");
            this.table.style.width = "100%";
            this.table.style.borderCollapse = "collapse";
            this.table.style.fontSize = "12px";
            this.scrollBox.appendChild(this.table);
            // 创建表头
            createTableHeader(this.table);
            // 固定底部居中按钮
            var btnWrap = document.createElement("div");
            btnWrap.style.position = "absolute";
            btnWrap.style.left = "50%";
            btnWrap.style.transform = "translateX(-50%)";
            btnWrap.style.zIndex = "10";
            btnWrap.style.display = "flex";
            var selectBtn = document.createElement("button");
            selectBtn.textContent = "📁 选择 LoRA";
            selectBtn.style.padding = "6px 14px";
            selectBtn.style.border = "none";
            selectBtn.style.borderRadius = "4px";
            selectBtn.style.background = "#4a86e8";
            selectBtn.style.color = "#fff";
            selectBtn.style.cursor = "pointer";
            selectBtn.onclick = function () {
                var self = this;
                if (!window.FxAiLoraSelector) {
                    alert("LoRA 选择器未加载");
                    return;
                }
                var currentNames = [];
                for (var i = 0; i < self.lines.length; i++) {
                    var n = self.lines[i].lora_name;
                    if (n) currentNames.push(n);
                }
                window.FxAiLoraSelector(currentNames.join(",")).then(function (res) {
                    console.log(res);
                    if (!res || !Array.isArray(res)) return;
                    for (var i = 0; i < res.length; i++) {
                        var item = res[i];
                        var targetName = (item.lora_name || "").trim();
                        if (!targetName) continue;
                        var exist = false;
                        for (var j = 0; j < self.lines.length; j++) {
                            if (self.lines[j].lora_name.trim() === targetName) {
                                exist = true;
                                break;
                            }
                        }
                        if (exist) continue;
                        addTableRow(self.table, self.lines,
                            item.lora_name || "",
                            item.model_strength || 1.0,
                            item.clip_strength || -1.0,
                            item.trigger_words || [],
                            item.enabled !== false,
                            item.invert || false,
                            item.fade_start || 1.0,
                            item.fade_end || 1.0
                        );
                        updateHiddenData(self);
                    }
                });
            }.bind(this);
            btnWrap.appendChild(selectBtn);
            this.wrap.appendChild(btnWrap);
            this.addDOMWidget("lora_wrap", "container", this.wrap);
            // 固定节点宽度
            this.size[0] = 920;
            this.setSize(this.computeSize());
            this.onResize = function (size) {
                if (size[0] < 920) this.size[0] = 920;
            };
            return r;
        };
        // 反序列化：仅解析对象格式，移除旧数组兼容
        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (!o || !o.widgets_values) return r;
            var data = null;
            for (var i = 0; i < o.widgets_values.length; i++) {
                var w = o.widgets_values[i];
                if (w.name === "lora_data") {
                    data = w.value;
                    break;
                }
            }
            if (!data) return r;
            try {
                var list = JSON.parse(data);
                if (Array.isArray(list)) {
                    clearTableRows(this.table, this.lines);
                    for (var j = 0; j < list.length; j++) {
                        var item = list[j];
                        var lora_name = item.lora_name || "";
                        var model_strength = item.model_strength || 1.0;
                        var clip_strength = item.clip_strength || -1.0;
                        var trigger_words = item.trigger_words || [];
                        var enabled = item.enabled !== false;
                        var invert = item.invert || false;
                        var fade_start = item.fade_start || 1.0;
                        var fade_end = item.fade_end || 1.0;

                        addTableRow(this.table, this.lines,
                            lora_name,
                            model_strength,
                            clip_strength,
                            trigger_words,
                            enabled,
                            invert,
                            fade_start,
                            fade_end
                        );
                    }
                }
            } catch (e) {}
            return r;
        };
        // 序列化：统一输出标准对象数组
        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];
            if (this.loraDataWidget) {
                var out = [];
                for (var i = 0; i < this.lines.length; i++) {
                    var line = this.lines[i];
                    out.push({
                        lora_name: line.lora_name,
                        model_strength: line.model_strength,
                        clip_strength: line.clip_strength,
                        trigger_words: line.trigger_words,
                        enabled: line.enabled,
                        invert: line.invert,
                        fade_start: line.fade_start,
                        fade_end: line.fade_end
                    });
                }
                var json = JSON.stringify(out);
                this.loraDataWidget.value = json;
                var found = false;
                for (var i = 0; i < o.widgets_values.length; i++) {
                    if (o.widgets_values[i].name === "lora_data") {
                        o.widgets_values[i].value = json;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    o.widgets_values.push({ name: "lora_data", value: json });
                }
            }
            return onSerialize ? onSerialize.apply(this, arguments) : o;
        };
    }
});
// 构建表头
function createTableHeader(table) {
    var thead = document.createElement("thead");
    var tr = document.createElement("tr");
    var cols = [
        { text: "序号", w: "40px" },
        { text: "LoRA名称" },
        { text: "权重", w: "35px" },
        { text: "CLIP", w: "35px" },
        { text: "触发词", w: "160px" },
        { text: "启用", w: "30px" },
        { text: "反转", w: "30px" },
        { text: "淡入", w: "30px" },
        { text: "淡出", w: "30px" },
        { text: "操作", w: "80px" }
    ];
    for (var i = 0; i < cols.length; i++) {
        var th = document.createElement("th");
        th.textContent = cols[i].text;
        th.style.width = cols[i].w;
        th.style.color = "#fff";
        th.style.textAlign = "center";
        th.style.padding = "4px 2px";
        tr.appendChild(th);
    }
    thead.appendChild(tr);
    table.appendChild(thead);
}
// 新增数据行
function addTableRow(table, lines, lora_name, model_strength, clip_strength, trigger_words, enabled, invert, fade_start, fade_end) {
    lora_name = lora_name || "";
    model_strength = model_strength || 1.0;
    clip_strength = clip_strength || -1.0;
    trigger_words = trigger_words || [];
    enabled = enabled !== false;
    invert = invert || false;
    fade_start = fade_start || 1.0;
    fade_end = fade_end || 1.0;
    var tr = document.createElement("tr");
    tr.style.height = "30px";
    // 序号
    var tdIdx = document.createElement("td");
    tdIdx.style.textAlign = "center";
    tdIdx.style.color = "#ccc";
    tdIdx.style.padding = "2px";
    tdIdx.textContent = lines.length + 1;
    tr.appendChild(tdIdx);
    // LoRA名称
    var tdName = document.createElement("td");
    tdName.style.padding = "2px";
    var nameInput = document.createElement("input");
    nameInput.style.width = "100%";
    nameInput.style.height = "24px";
    nameInput.style.boxSizing = "border-box";
    nameInput.value = lora_name;
    tdName.appendChild(nameInput);
    tr.appendChild(tdName);
    // 模型权重
    var tdModel = document.createElement("td");
    tdModel.style.padding = "2px";
    var modelInput = document.createElement("input");
    modelInput.type = "text";
    modelInput.step = "0.1";
    modelInput.style.width = "100%";
    modelInput.style.height = "24px";
    modelInput.style.boxSizing = "border-box";
    modelInput.value = model_strength;
    tdModel.appendChild(modelInput);
    tr.appendChild(tdModel);
    // CLIP权重
    var tdClip = document.createElement("td");
    tdClip.style.padding = "2px";
    var clipInput = document.createElement("input");
    clipInput.type = "text";
    clipInput.step = "0.1";
    clipInput.style.width = "100%";
    clipInput.style.height = "24px";
    clipInput.style.boxSizing = "border-box";
    clipInput.value = clip_strength;
    tdClip.appendChild(clipInput);
    tr.appendChild(tdClip);
    // 触发词
    var tdTrigger = document.createElement("td");
    tdTrigger.style.padding = "2px";
    var triggerInput = document.createElement("input");
    triggerInput.style.width = "100%";
    triggerInput.style.height = "24px";
    triggerInput.style.boxSizing = "border-box";
    triggerInput.value = Array.isArray(trigger_words) ? trigger_words.join(", ") : trigger_words;
    tdTrigger.appendChild(triggerInput);
    tr.appendChild(tdTrigger);
    // 启用
    var tdEnable = document.createElement("td");
    tdEnable.style.textAlign = "center";
    tdEnable.style.padding = "2px";
    var enableCheck = document.createElement("input");
    enableCheck.type = "checkbox";
    enableCheck.checked = enabled;
    tdEnable.appendChild(enableCheck);
    tr.appendChild(tdEnable);
    // 反转
    var tdInvert = document.createElement("td");
    tdInvert.style.textAlign = "center";
    tdInvert.style.padding = "2px";
    var invertCheck = document.createElement("input");
    invertCheck.type = "checkbox";
    invertCheck.checked = invert;
    tdInvert.appendChild(invertCheck);
    tr.appendChild(tdInvert);
    // 淡入
    var tdFadeIn = document.createElement("td");
    tdFadeIn.style.padding = "2px";
    var fadeInInput = document.createElement("input");
    fadeInInput.type = "text";
    fadeInInput.step = "0.1";
    fadeInInput.style.width = "100%";
    fadeInInput.style.height = "24px";
    fadeInInput.style.boxSizing = "border-box";
    fadeInInput.value = fade_start;
    tdFadeIn.appendChild(fadeInInput);
    tr.appendChild(tdFadeIn);
    // 淡出
    var tdFadeOut = document.createElement("td");
    tdFadeOut.style.padding = "2px";
    var fadeOutInput = document.createElement("input");
    fadeOutInput.type = "text";
    fadeOutInput.step = "0.1";
    fadeOutInput.style.width = "100%";
    fadeOutInput.style.height = "24px";
    fadeOutInput.style.boxSizing = "border-box";
    fadeOutInput.value = fade_end;
    tdFadeOut.appendChild(fadeOutInput);
    tr.appendChild(tdFadeOut);
    // 操作按钮
    var tdBtn = document.createElement("td");
    tdBtn.style.textAlign = "center";
    tdBtn.style.padding = "2px";
    tdBtn.style.display = "flex";
    tdBtn.style.gap = "4px";
    tdBtn.style.justifyContent = "center";
    var upBtn = document.createElement("button");
    upBtn.textContent = "↑";
    upBtn.style.width = "24px";
    upBtn.style.height = "24px";
    upBtn.style.border = "none";
    upBtn.style.background = "#4a86e8";
    upBtn.style.color = "#fff";
    upBtn.style.cursor = "pointer";
    var downBtn = document.createElement("button");
    downBtn.textContent = "↓";
    downBtn.style.width = "24px";
    downBtn.style.height = "24px";
    downBtn.style.border = "none";
    downBtn.style.background = "#4a86e8";
    downBtn.style.color = "#fff";
    downBtn.style.cursor = "pointer";
    var delBtn = document.createElement("button");
    delBtn.textContent = "删";
    delBtn.style.width = "24px";
    delBtn.style.height = "24px";
    delBtn.style.border = "none";
    delBtn.style.background = "#c52222";
    delBtn.style.color = "#fff";
    delBtn.style.cursor = "pointer";
    tdBtn.appendChild(upBtn);
    tdBtn.appendChild(downBtn);
    tdBtn.appendChild(delBtn);
    tr.appendChild(tdBtn);
    // 行数据对象
    var lineData = {
        tr: tr,
        tdIdx: tdIdx,
        lora_name: lora_name,
        model_strength: model_strength,
        clip_strength: clip_strength,
        trigger_words: trigger_words,
        enabled: enabled,
        invert: invert,
        fade_start: fade_start,
        fade_end: fade_end
    };
    lines.push(lineData);
    // 数据同步
    function syncData() {
        lineData.lora_name = nameInput.value.trim();
        lineData.model_strength = parseFloat(modelInput.value) || 1.0;
        lineData.clip_strength = parseFloat(clipInput.value) || -1.0;
        var trigs = triggerInput.value.split(",").map(function (t) {
            return t.trim();
        }).filter(function (t) {
            return t;
        });
        lineData.trigger_words = trigs;
        lineData.enabled = enableCheck.checked;
        lineData.invert = invertCheck.checked;
        lineData.fade_start = parseFloat(fadeInInput.value) || 1.0;
        lineData.fade_end = parseFloat(fadeOutInput.value) || 1.0;
        updateHiddenData(table.closest(".comfyui-widget-container").widget);
    }
    nameInput.oninput = syncData;
    modelInput.oninput = syncData;
    clipInput.oninput = syncData;
    triggerInput.oninput = syncData;
    enableCheck.onchange = syncData;
    invertCheck.onchange = syncData;
    fadeInInput.oninput = syncData;
    fadeOutInput.oninput = syncData;
    // 上下移动
    upBtn.onclick = function () {
        moveRow(table, lines, lineData, -1);
    };
    downBtn.onclick = function () {
        moveRow(table, lines, lineData, 1);
    };
    // 删除
    delBtn.onclick = function () {
        removeRow(table, lines, lineData);
        updateHiddenData(table.closest(".comfyui-widget-container").widget);
    };
    var tbody = table.querySelector("tbody") || document.createElement("tbody");
    if (!table.querySelector("tbody")) {
        table.appendChild(tbody);
    }
    tbody.appendChild(tr);
}
// 清空所有数据行
function clearTableRows(table, lines) {
    var tbody = table.querySelector("tbody");
    if (tbody) tbody.innerHTML = "";
    lines.length = 0;
}
// 行上下移动
function moveRow(table, lines, lineData, dir) {
    var idx = -1;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i] === lineData) {
            idx = i;
            break;
        }
    }
    if (idx === -1) return;
    var newIdx = idx + dir;
    if (newIdx < 0 || newIdx >= lines.length) return;
    var temp = lines[idx];
    lines[idx] = lines[newIdx];
    lines[newIdx] = temp;
    var tbody = table.querySelector("tbody");
    var allTr = tbody.querySelectorAll("tr");
    if (dir === -1) {
        tbody.insertBefore(allTr[idx], allTr[newIdx]);
    } else {
        tbody.insertBefore(allTr[idx], allTr[newIdx].nextSibling);
    }
    refreshIndex(lines);
}
// 刷新序号
function refreshIndex(lines) {
    for (var i = 0; i < lines.length; i++) {
        lines[i].tdIdx.textContent = i + 1;
    }
}
// 删除单行
function removeRow(table, lines, lineData) {
    lineData.tr.remove();
    for (var i = 0; i < lines.length; i++) {
        if (lines[i] === lineData) {
            lines.splice(i, 1);
            break;
        }
    }
    refreshIndex(lines);
}
// 更新隐藏字段数据（纯对象格式）
function updateHiddenData(node) {
    if (!node || !node.loraDataWidget) return;
    var out = [];
    for (var i = 0; i < node.lines.length; i++) {
        var d = node.lines[i];
        out.push({
            lora_name: d.lora_name,
            model_strength: d.model_strength,
            clip_strength: d.clip_strength,
            trigger_words: d.trigger_words,
            enabled: d.enabled,
            invert: d.invert,
            fade_start: d.fade_start,
            fade_end: d.fade_end
        });
    }
    node.loraDataWidget.value = JSON.stringify(out);
}