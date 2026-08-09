import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ==============================================
// 短剧角色资源管理器：行内编辑 + 持久化到 fxai.db
// 字段：name / avatar(单图) / voice / description
// ==============================================

function apiURL(path) {
    return api.apiURL(path);
}

function getCharacterList() {
    return fetch(apiURL("/fxai/characters/list"))
        .then(function (r) { return r.json(); })
        .then(function (d) { return d.characters || []; })
        .catch(function () { return []; });
}

function saveCharactersBatch(list) {
    return fetch(apiURL("/fxai/characters/save_batch"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(list)
    }).then(function (r) { return r.json(); });
}

app.registerExtension({
    name: "FxAiCharacterProfileManager",
    beforeRegisterNodeDef: function (nodeType, nodeData) {
        if (nodeData.name !== "FxAiCharacterProfileManager") return;

        var onNodeCreated = nodeType.prototype.onNodeCreated;
        var onConfigure = nodeType.prototype.onConfigure;
        var onSerialize = nodeType.prototype.onSerialize;

        nodeType.prototype.onNodeCreated = function () {
            var r = onNodeCreated ? onNodeCreated.apply(this, arguments) : void 0;
            this.rows = [];

            this.dataWidget = null;
            for (var i = 0; i < this.widgets.length; i++) {
                var w = this.widgets[i];
                if (w && w.name === "角色数据") {
                    this.dataWidget = w;
                    setTimeout(function () { w.hidden = true; }, 0);
                    break;
                }
            }

            this.scrollContainer = document.createElement("div");
            this.scrollContainer.style.height = "100%";
            this.scrollContainer.style.overflowY = "auto";
            this.scrollContainer.style.overflowX = "hidden";
            this.scrollContainer.style.minWidth = "760px";
            this.scrollContainer.style.margin = "5px 0";
            this.scrollContainer.style.paddingRight = "5px";
            this.scrollContainer.style.boxSizing = "border-box";
            this.addDOMWidget("characters_container", "container", this.scrollContainer);

            createHeader(this);

            var self = this;
            this.addWidget("button", "➕ 新增角色", null, function () {
                addRow(self, null, {});
            });
            this.addWidget("button", "📚 从数据库加载", null, function () {
                self.loadFromDB();
            });
            this.addWidget("button", "💾 保存到数据库", null, function () {
                self.saveAllToDB();
            });

            var FIXED_WIDTH = 820;
            this.size[0] = FIXED_WIDTH;
            this.setSize(this.computeSize());
            this.onResize = function (size) {
                if (size[0] < FIXED_WIDTH) {
                    this.size[0] = FIXED_WIDTH;
                    this.setSize([FIXED_WIDTH, size[1]]);
                }
            };

            setTimeout(function () {
                if (self.rows.length === 0) self.loadFromDB();
            }, 50);

            return r;
        };

        nodeType.prototype.onConfigure = function (o) {
            var r = onConfigure ? onConfigure.apply(this, arguments) : void 0;
            var self = this;
            getCharacterList().then(function (chars) {
                clearRows(self);
                chars.forEach(function (c) { addRow(self, null, c); });
                refreshLineNumbers(self);
                updateHidden(self);
            });
            return r;
        };

        nodeType.prototype.onSerialize = function (o) {
            o = o || {};
            o.widgets_values = o.widgets_values || [];
            if (this.dataWidget && this.rows) {
                var json = JSON.stringify(collectData(this));
                this.dataWidget.value = json;
                var found = false;
                for (var i = 0; i < o.widgets_values.length; i++) {
                    var w = o.widgets_values[i];
                    if (w && w.name === "角色数据") {
                        w.value = json;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    o.widgets_values.push({ name: "角色数据", value: json });
                }
            }
            return onSerialize ? onSerialize.apply(this, arguments) : o;
        };

        nodeType.prototype.loadFromDB = function () {
            var self = this;
            getCharacterList().then(function (chars) {
                if (chars.length === 0) {
                    if (self.rows.length === 0) addRow(self, null, {});
                    updateHidden(self);
                    refreshLineNumbers(self);
                    return;
                }
                clearRows(self);
                chars.forEach(function (c) { addRow(self, null, c); });
                refreshLineNumbers(self);
                updateHidden(self);
            });
        };

        nodeType.prototype.saveAllToDB = function () {
            var d = collectData(this);
            if (!d.length) return;
            saveCharactersBatch(d).then(function (res) {
                if (!res || !res.saved) return;
                var rows = this.rows;
                for (var i = 0; i < res.saved.length && i < rows.length; i++) {
                    if (rows[i].id === null) rows[i].id = res.saved[i].id;
                }
                updateHidden(this);
            }.bind(this)).catch(function () {});
        };
    }
});

function addRow(node, null2, data) {
    addRowFrom(node, data);
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
    header.style.color = "#fff";

    var labels = [
        { text: "序号", width: "34px" },
        { text: "角色名称", width: "110px" },
        { text: "形象照", width: "100px" },
        { text: "音色", width: "150px" },
        { text: "角色描述", flex: 1 },
        { text: "操作", width: "34px" }
    ];
    for (var m = 0; m < labels.length; m++) {
        var it = labels[m];
        var span = document.createElement("span");
        span.textContent = it.text;
        span.style.textAlign = "center";
        if (it.width) span.style.minWidth = it.width;
        if (it.flex) span.style.flex = it.flex;
        span.style.flexShrink = "0";
        header.appendChild(span);
    }
    node.scrollContainer.appendChild(header);
}

function collectData(node) {
    return (node.rows || []).map(function (item) {
        return {
            id: item.id || null,
            name: item.name || "",
            avatar: (item.avatar || "").split(",")[0] || "",
            voice: item.voice || "",
            description: item.description || ""
        };
    }).filter(function (x) {
        return (x.name || "").trim() !== "" ||
               (x.avatar || "").trim() !== "" ||
               (x.voice || "").trim() !== "" ||
               (x.description || "").trim() !== "";
    });
}

function clearRows(node) {
    node.rows.forEach(function (item) { item.row.remove(); });
    node.rows = [];
}

function addRowFrom(node, data) {
    var d = data || {};
    var id = d.id || null;
    var name = d.name || "";
    var avatar = d.avatar || "";
    if (!avatar && d.images) avatar = String(d.images).split(",")[0] || "";
    var voice = d.voice || "";
    var description = d.description || "";

    var idx = node.rows.length;
    var row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "flex-start";
    row.style.gap = "6px";
    row.style.width = "100%";
    row.style.marginBottom = "8px";
    row.style.boxSizing = "border-box";

    var lineNumLabel = document.createElement("span");
    lineNumLabel.textContent = (idx + 1) + ".";
    lineNumLabel.style.minWidth = "34px";
    lineNumLabel.style.textAlign = "center";
    lineNumLabel.style.color = "var(--fg-color)";
    lineNumLabel.style.opacity = "0.7";
    lineNumLabel.style.fontFamily = "monospace";
    lineNumLabel.style.fontSize = "12px";
    lineNumLabel.style.marginTop = "6px";
    lineNumLabel.style.flexShrink = "0";

    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.placeholder = "角色名";
    nameInput.style.width = "110px";
    nameInput.style.height = "28px";
    nameInput.style.padding = "0 6px";
    nameInput.style.borderRadius = "4px";
    nameInput.style.border = "1px solid var(--comfy-menu-border-color)";
    nameInput.style.backgroundColor = "var(--comfy-input-bg)";
    nameInput.style.color = "var(--fg-color)";
    nameInput.style.flexShrink = "0";
    nameInput.value = name;

    var avatarBox = document.createElement("div");
    avatarBox.style.width = "100px";
    avatarBox.style.height = "70px";
    avatarBox.style.padding = "3px";
    avatarBox.style.borderRadius = "4px";
    avatarBox.style.border = "1px dashed var(--comfy-menu-border-color)";
    avatarBox.style.backgroundColor = "var(--comfy-input-bg)";
    avatarBox.style.boxSizing = "border-box";
    avatarBox.style.cursor = "pointer";
    avatarBox.style.display = "flex";
    avatarBox.style.alignItems = "center";
    avatarBox.style.justifyContent = "center";
    avatarBox.style.flexShrink = "0";
    avatarBox.title = "点击选择形象照（单张，再次点击可更换）";
    avatarBox.onclick = function () {
        openAvatarSelector(item);
    };

    var voiceBox = document.createElement("div");
    voiceBox.style.width = "150px";
    voiceBox.style.minHeight = "34px";
    voiceBox.style.padding = "3px 6px";
    voiceBox.style.borderRadius = "4px";
    voiceBox.style.border = "1px dashed var(--comfy-menu-border-color)";
    voiceBox.style.backgroundColor = "var(--comfy-input-bg)";
    voiceBox.style.boxSizing = "border-box";
    voiceBox.style.cursor = "pointer";
    voiceBox.style.fontSize = "11px";
    voiceBox.style.color = "#bbb";
    voiceBox.style.flexShrink = "0";
    voiceBox.style.lineHeight = "20px";
    voiceBox.style.marginTop = "2px";
    voiceBox.style.display = "flex";
    voiceBox.style.flexDirection = "column";
    voiceBox.style.alignItems = "stretch";
    voiceBox.title = "点击选择角色音色（可选，不配则系统默认生成）";
    voiceBox.onclick = function (e) {
        if (e.target && e.target.tagName === "AUDIO") return;
        if (e.target && e.target.classList && (e.target.classList.contains("vchg") || e.target.classList.contains("vdel"))) return;
        openVoiceSelector(item);
    };

    var descInput = document.createElement("textarea");
    descInput.placeholder = "角色描述：外貌/服装/性格/身份等";
    descInput.style.flex = "1";
    descInput.style.minWidth = "0";
    descInput.style.minHeight = "52px";
    descInput.style.padding = "4px 6px";
    descInput.style.borderRadius = "4px";
    descInput.style.fontFamily = "monospace";
    descInput.style.fontSize = "11px";
    descInput.style.border = "1px solid var(--comfy-menu-border-color)";
    descInput.style.backgroundColor = "var(--comfy-input-bg)";
    descInput.style.color = "var(--fg-color)";
    descInput.style.resize = "vertical";
    descInput.style.boxSizing = "border-box";
    descInput.value = description;

    var delBtn = mkBtn("✕", "#c52222", "删除");

    row.appendChild(lineNumLabel);
    row.appendChild(nameInput);
    row.appendChild(avatarBox);
    row.appendChild(voiceBox);
    row.appendChild(descInput);
    row.appendChild(delBtn);
    node.scrollContainer.appendChild(row);

    var item = {
        node: node,
        row: row,
        nameInput: nameInput,
        avatarBox: avatarBox,
        voiceBox: voiceBox,
        descInput: descInput,
        label: lineNumLabel,
        id: id,
        name: name,
        avatar: avatar,
        voice: voice,
        description: description
    };
    node.rows.push(item);

    renderAvatarBox(item);
    renderVoiceBox(item);

    nameInput.addEventListener("input", function () {
        item.name = nameInput.value;
        updateHidden(node);
    });
    descInput.addEventListener("input", function () {
        item.description = descInput.value;
        updateHidden(node);
    });
    delBtn.addEventListener("click", function () { removeRow(node, item); });

    setTimeout(function () {
        node.scrollContainer.scrollTop = node.scrollContainer.scrollHeight;
    }, 10);
    updateHidden(node);
}

function mkBtn(text, bg, title) {
    var b = document.createElement("button");
    b.textContent = text;
    b.title = title;
    b.style.width = "25px";
    b.style.height = "25px";
    b.style.borderRadius = "4px";
    b.style.border = "none";
    b.style.cursor = "pointer";
    b.style.fontWeight = "bold";
    b.style.backgroundColor = bg;
    b.style.color = "#fff";
    b.style.flexShrink = "0";
    b.style.marginTop = "2px";
    return b;
}

function renderAvatarBox(item) {
    var box = item.avatarBox;
    box.innerHTML = "";
    var avatar = (item.avatar || "").split(",")[0] || "";
    if (!avatar) {
        var tip = document.createElement("span");
        tip.textContent = "＋ 选形象图";
        tip.style.color = "#999";
        tip.style.fontSize = "11px";
        box.appendChild(tip);
        return;
    }
    var parts = avatar.split("/");
    var sub = parts[0] || "";
    var fname = parts[1] || "";
    var wrap = document.createElement("div");
    wrap.style.position = "relative";
    wrap.style.width = "90px";
    wrap.style.height = "60px";
    wrap.style.borderRadius = "4px";
    wrap.style.overflow = "hidden";
    wrap.style.flexShrink = "0";
    var img = document.createElement("img");
    img.style.width = "100%";
    img.style.height = "100%";
    img.style.objectFit = "cover";
    img.style.display = "block";
    img.onerror = function () { img.style.opacity = "0.3"; };
    if (sub && fname) {
        img.src = apiURL("/fxai/image/v2/preview?subdir=" + encodeURIComponent(sub) + "&filename=" + encodeURIComponent(fname));
    }
    wrap.appendChild(img);
    box.appendChild(wrap);
}

function renderVoiceBox(item) {
    var box = item.voiceBox;
    box.innerHTML = "";
    if (!item.voice) {
        var tip = document.createElement("span");
        tip.textContent = "🎵 点击选音色（可留空）";
        tip.style.cssText = "display:block;padding:6px 2px;";
        box.appendChild(tip);
        box.style.padding = "3px 6px";
        box.style.backgroundColor = "var(--comfy-input-bg)";
        box.style.border = "1px dashed var(--comfy-menu-border-color)";
        return;
    }
    var raw = item.voice || "";
    var rel = raw.replace(/^\/fxai\/audio\//, "").replace(/^fxai\/audio\//, "");
    var parts = rel.split("/");
    var sub = parts.length > 1 ? parts[0] : "";
    var fname = parts[parts.length - 1] || rel;
var audio = document.createElement("audio");
    audio.src = apiURL("/fxai/audio/preview?subdir=" + encodeURIComponent(sub) + "&filename=" + encodeURIComponent(fname));
    audio.controls = true;
    audio.preload = "none";
    audio.style.cssText = "width:100%;display:block;height:30px;";
    var chg = document.createElement("span");
    chg.textContent = "🔁 点击更换音色";
    chg.className = "vchg";
    chg.style.cssText = "display:block;text-align:center;color:#4a8fe8;font-size:11px;cursor:pointer;padding:1px 0;line-height:14px;user-select:none;";
    chg.onclick = function (e) {
        e.stopPropagation();
        openVoiceSelector(item);
    };
    var del = document.createElement("span");
    del.textContent = "✕ 清除";
    del.className = "vdel";
    del.style.cssText = "display:block;text-align:center;color:#f54242;font-size:10px;cursor:pointer;padding:0 0 2px;line-height:14px;user-select:none;";
    del.title = "删除该音色（不配音）";
    del.onclick = function (e) {
        e.stopPropagation();
        item.voice = "";
        renderVoiceBox(item);
        updateHidden(item.node);
    };
    box.style.padding = "0";
    box.style.backgroundColor = "transparent";
    box.style.border = "none";
    box.appendChild(audio);
    box.appendChild(chg);
    box.appendChild(del);
    box.title = item.voice;
}

function openAvatarSelector(item) {
    FxAiCharacterAssetsSelector(item.avatar).then(function (val) {
        if (val !== undefined) {
            var first = String(val).split(",")[0] || "";
            item.avatar = first;
            renderAvatarBox(item);
            updateHidden(item.node);
        }
    });
}

function openVoiceSelector(item) {
    FxAiAudioSelector(item.voice).then(function (val) {
        if (val !== undefined) {
            item.voice = String(val).replace(/^\/fxai\/audio\//, "");
            renderVoiceBox(item);
            updateHidden(item.node);
        }
    });
}

function updateHidden(node) {
    if (!node || !node.dataWidget) return;
    var arr = collectData(node);
    node.dataWidget.value = JSON.stringify(arr);
    if (node.dataWidget.inputEl) {
        node.dataWidget.inputEl.value = JSON.stringify(arr);
        var ev = document.createEvent("Event");
        ev.initEvent("input", true, true);
        node.dataWidget.inputEl.dispatchEvent(ev);
    }
}

function removeRow(node, item) {
    if (item.id) {
        fetch(apiURL("/fxai/characters/delete"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: item.id })
        }).catch(function () {});
    }
    item.row.remove();
    node.rows = node.rows.filter(function (x) { return x !== item; });
    refreshLineNumbers(node);
    updateHidden(node);
}

function refreshLineNumbers(node) {
    for (var i = 0; i < node.rows.length; i++) node.rows[i].label.textContent = (i + 1) + ".";
}