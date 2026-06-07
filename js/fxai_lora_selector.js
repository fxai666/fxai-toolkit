import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ==============================================
// 核心：获取Lora列表（后端返回对象格式 { lora名: 配置 }）
// ==============================================
function fetchLoraFileList() {
    return new Promise(function(resolve) {
        var url = api.apiURL("/fxai/lora/files");
        fetch(url)
        .then(function(resp) {
            return resp.ok ? resp.json() : {};
        })
        .then(function(data) {
            var list = [];
            for (var loraName in data) {
                if (data.hasOwnProperty(loraName)) {
                    list.push({
                        lora_name: loraName,
                        config: data[loraName] || {}
                    });
                }
            }
            list.sort(function(a, b) {
                return a.lora_name.toLowerCase().localeCompare(b.lora_name.toLowerCase());
            });
            resolve(list);
        })
        .catch(function() {
            resolve([]);
        });
    });
}

// ==============================================
// Lora选择器（表格版 ES5）
// 传入：selectedStr 逗号分隔文件名，用于默认选中
// 返回：选中的配置数组 [ {lora_name,...config}, ... ]
// ==============================================
window.FxAiLoraSelector = function(selectedStr) {
    return new Promise(function(resolve) {
        var selectedItems = [];
        var currentList = [];
        var defaultSelectedNames = [];

        // 解析传入的默认选中项（逗号分隔）
        if (selectedStr && typeof selectedStr === "string") {
            defaultSelectedNames = selectedStr.split(",").map(function(item) {
                return item.trim();
            }).filter(function(item) {
                return item;
            });
        }

        // 遮罩层
        var mask = document.createElement("div");
        mask.style.cssText = "position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.85);display:flex;align-items:center;justify-content:center;";
        document.body.appendChild(mask);

        // 弹窗
        var modal = document.createElement("div");
        modal.style.cssText = "width:1200px;max-width:95vw;height:600px;max-height:90vh;background:#222;border-radius:10px;padding:20px;box-sizing:border-box;display:flex;flex-direction:column;gap:16px;";
        mask.appendChild(modal);

        // 标题栏
        var header = document.createElement("div");
        header.style.cssText = "display:flex;justify-content:space-between;align-items:center;";
        modal.appendChild(header);

        var title = document.createElement("div");
        title.textContent = "🧩 LoRA 选择器";
        title.style.cssText = "font-size:18px;color:#fff;font-weight:bold;";
        header.appendChild(title);

        var selectedTip = document.createElement("div");
        selectedTip.style.cssText = "font-size:12px;color:#999;";
        selectedTip.textContent = "已选中：0 个";
        header.appendChild(selectedTip);

        // 表格容器
        var tableContainer = document.createElement("div");
        tableContainer.style.cssText = "flex:1;overflow-y:auto;background:#2b2b2b;border-radius:6px;padding:4px;";
        modal.appendChild(tableContainer);

        // 底部按钮
        var bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display:flex;justify-content:flex-end;gap:10px;";
        modal.appendChild(bottomBar);

        var btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        btnCancel.style.cssText = "padding:6px 16px;border:none;border-radius:4px;background:#555;color:#fff;cursor:pointer;";
        btnCancel.onclick = function() {
            resolve(null);
            closeModal();
        };

        var btnConfirm = document.createElement("button");
        btnConfirm.textContent = "✅ 确认选择";
        btnConfirm.style.cssText = "padding:6px 16px;border:none;border-radius:4px;background:#4a8fff;color:#fff;cursor:pointer;";
        btnConfirm.onclick = function() {
            resolve(selectedItems);
            closeModal();
        };

        bottomBar.appendChild(btnCancel);
        bottomBar.appendChild(btnConfirm);

        // 关闭
        function closeModal() {
            document.body.removeChild(mask);
        }

        mask.onclick = function(e) {
            if (e.target === mask) closeModal();
        };

        // 更新选中数量
        function updateSelectedTip() {
            selectedTip.textContent = "已选中：" + selectedItems.length + " 个 LoRA";
        }

        // 渲染表格
        function renderTable() {
            fetchLoraFileList().then(function(list) {
                currentList = list;
                tableContainer.innerHTML = "";

                if (!list || list.length === 0) {
                    tableContainer.innerHTML = "<div style=\"color:#999;text-align:center;padding:30px;\">暂无 LoRA 文件</div>";
                    return;
                }

                var table = document.createElement("table");
                table.style.width = "100%";
                table.style.borderCollapse = "collapse";
                table.style.color = "#fff";
                table.style.fontSize = "14px";

                var thead = document.createElement("thead");
                thead.innerHTML = "<tr style=\"background:#3a3a3a;text-align:left;\">" +
                    "<th style=\"padding:10px;width:50px;\">选择</th>" +
                    "<th style=\"padding:10px;\">LoRA 文件名</th>" +
                    "<th style=\"padding:10px;\">功能描述</th>" +
                    "<th style=\"padding:10px;\">触发词</th>" +
                "</tr>";
                table.appendChild(thead);

                var tbody = document.createElement("tbody");
                table.appendChild(tbody);

                for (var i = 0; i < list.length; i++) {
                    var item = list[i];
                    var lora_name = item.lora_name;
                    var config = item.config || {};

                    var desc = config.desc || config.description || "";
                    var triggers = Array.isArray(config.trigger_words)
                        ? config.trigger_words.join(", ")
                        : (config.trigger_words || "");

                    var tr = document.createElement("tr");
                    tr.style.borderBottom = "1px solid #444";
                    tr.style.cursor = "pointer";

                    // 判断是否默认选中
                    var isChecked = defaultSelectedNames.indexOf(lora_name) !== -1;
                    var checkedAttr = isChecked ? "checked" : "";

                    tr.innerHTML =
                        "<td style=\"padding:10px;text-align:center;\">" +
                            "<input type=\"checkbox\" class=\"lora-check\" data-index=\"" + i + "\" " + checkedAttr + ">" +
                        "</td>" +
                        "<td style=\"padding:10px;\">" + lora_name + "</td>" +
                        "<td style=\"padding:10px;color:#aaa;\">" + desc + "</td>" +
                        "<td style=\"padding:10px;color:#4a8fff;\">" + (triggers || "-") + "</td>";

                    tr.onclick = function(e) {
                        if (e.target.tagName === "INPUT") return;
                        var checkbox = this.querySelector(".lora-check");
                        checkbox.checked = !checkbox.checked;
                        onCheckChange();
                    };

                    var checkbox = tr.querySelector(".lora-check");
                    checkbox.onchange = onCheckChange;
                    tbody.appendChild(tr);
                }

                tableContainer.appendChild(table);
                onCheckChange();
                updateSelectedTip();
            });
        }

        // 选中变更
        function onCheckChange() {
            selectedItems = [];
            var checks = document.querySelectorAll(".lora-check");
            for (var i = 0; i < checks.length; i++) {
                var chk = checks[i];
                if (chk.checked) {
                    var idx = parseInt(chk.getAttribute("data-index"));
                    var data = currentList[idx];
                    var obj = {
                        lora_name: data.lora_name
                    };
                    for (var key in data.config) {
                        if (data.config.hasOwnProperty(key)) {
                            obj[key] = data.config[key];
                        }
                    }
                    selectedItems.push(obj);
                }
            }
            updateSelectedTip();
        }

        renderTable();
    });
};

app.registerExtension({
    name: "FxAiLoraSelector"
});