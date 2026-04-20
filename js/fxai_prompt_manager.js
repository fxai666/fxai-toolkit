import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiPromptManager";

// 获取提示词文件列表
async function fetchFileList(subdir) {
    const resp = await fetch(api.apiURL(`/fxai/prompt/list?subdir=${encodeURIComponent(subdir)}`));
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.files;
}

// 删除提示词文件
async function deletePrompt(subdir, filename) {
    try {
        const resp = await fetch(api.apiURL(`/fxai/prompt/delete?subdir=${encodeURIComponent(subdir)}&filename=${encodeURIComponent(filename)}`));
        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.error || "删除失败");
        }
        return true;
    } catch (err) {
        alert("删除失败：" + err.message);
        return false;
    }
}

// 保存前端自定义输入的提示词
async function saveManualPrompt(subdir, filename, content) {
    try {
        const formData = new FormData();
        formData.append("subdir", subdir);
        formData.append("filename", filename);
        formData.append("content", content);
        
        const response = await fetch(api.apiURL("/fxai/prompt/save_manual"), {
            method: "POST",
            body: formData,
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "保存失败");
        }
        return true;
    } catch (err) {
        alert("保存失败：" + err.message);
        return false;
    }
}

// 构建自定义UI
function addUI(node) {
    if (node._uiAdded) return;
    node._uiAdded = true;

    const subdirWidget = node.widgets.find(w => w.name === "目录");
    if (!subdirWidget) return;

    // 主容器
    const container = document.createElement("div");
    container.style.padding = "8px";
    container.style.border = "1px solid #555";
    container.style.borderRadius = "4px";
    container.style.minWidth = "380px";
    node.addDOMWidget("prompt_ui", "prompt_ui", container, { serialize: false });

    // ======================
    // 自定义：手动输入提示词区域
    // ======================
    const manualPromptWrap = document.createElement("div");
    manualPromptWrap.style.marginBottom = "12px";
    
    // 文件名输入行
    const nameWrap = document.createElement("div");
    nameWrap.style.display = "flex";
    nameWrap.style.gap = "8px";
    nameWrap.style.marginBottom = "8px";
    nameWrap.style.alignItems = "center";

    const nameLabel = document.createElement("div");
    nameLabel.textContent = "保存文件名：";
    nameLabel.style.color = "#ccc";

    const filenameInput = document.createElement("input");
    filenameInput.type = "text";
    filenameInput.placeholder = "请输入文件名（无需.txt后缀）";
    filenameInput.style.flex = 1;
    filenameInput.style.padding = "5px";
    filenameInput.style.backgroundColor = "#2a2a2a";
    filenameInput.style.color = "#fff";
    filenameInput.style.border = "1px solid #666";
    filenameInput.style.borderRadius = "4px";

    nameWrap.appendChild(nameLabel);
    nameWrap.appendChild(filenameInput);
    manualPromptWrap.appendChild(nameWrap);

    // 大提示词多行输入框
    const promptTextarea = document.createElement("textarea");
    promptTextarea.placeholder = "手动输入提示词";
    promptTextarea.style.width = "100%";
    promptTextarea.style.minHeight = "120px";
    promptTextarea.style.padding = "8px";
    promptTextarea.style.backgroundColor = "#2a2a2a";
    promptTextarea.style.color = "#fff";
    promptTextarea.style.border = "1px solid #666";
    promptTextarea.style.borderRadius = "4px";
    promptTextarea.style.resize = "vertical";
    manualPromptWrap.appendChild(promptTextarea);

    // 保存+刷新按钮行（合并到同一行）
    const btnRow = document.createElement("div");
    btnRow.style.display = "flex";
    btnRow.style.gap = "8px";
    btnRow.style.marginTop = "8px";

    // 【确认保存】按钮
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "✅ 确认保存";
    saveBtn.style.padding = "6px 12px";
    saveBtn.style.backgroundColor = "#2d7d46";
    saveBtn.style.color = "#fff";
    saveBtn.style.border = "none";
    saveBtn.style.borderRadius = "4px";
    saveBtn.style.cursor = "pointer";
    
    saveBtn.onclick = async function() {
        const filename = filenameInput.value.trim();
        const content = promptTextarea.value.trim();
        
        if (!filename) {
            alert("请输入保存的文件名！");
            return;
        }
        if (!content) {
            alert("请输入提示词内容！");
            return;
        }

        saveBtn.disabled = true;
        saveBtn.textContent = "保存中...";
        try {
            const isSuccess = await saveManualPrompt(
                subdirWidget.value,
                filename,
                content
            );
            if (isSuccess) {
                filenameInput.value = "";
                promptTextarea.value = "";
                await updateList(); // 保存成功自动刷新列表
            }
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = "✅ 确认保存";
        }
    };

    // 刷新按钮
    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄 刷新列表";
    refreshBtn.style.padding = "6px 12px";
    refreshBtn.style.backgroundColor = "#444";
    refreshBtn.style.color = "#fff";
    refreshBtn.style.border = "none";
    refreshBtn.style.borderRadius = "4px";
    refreshBtn.style.cursor = "pointer";
    refreshBtn.onclick = updateList;

    // 将两个按钮添加到同一行
    btnRow.appendChild(saveBtn);
    btnRow.appendChild(refreshBtn);
    manualPromptWrap.appendChild(btnRow);
    container.appendChild(manualPromptWrap);

    // 提示词文件列表容器
    const listWrap = document.createElement("div");
    listWrap.style.maxHeight = "400px";
    listWrap.style.overflowY = "auto";
    listWrap.style.border = "1px solid #666";
    listWrap.style.padding = "4px";
    container.appendChild(listWrap);

    // ========== 列表渲染：严格 1.2.3.4 数字序号 + 文件名 ==========
    async function updateList() {
        const files = await fetchFileList(subdirWidget.value);
        listWrap.innerHTML = "";

        files.forEach(function(file, index) {
            const item = document.createElement("div");
            item.style.padding = "6px 8px";
            item.style.backgroundColor = "#222";
            item.style.borderRadius = "4px";
            item.style.marginBottom = "4px";
            item.style.display = "flex";
            item.style.justifyContent = "space-between";
            item.style.alignItems = "center";

            // 序号 + 文件名主体
            const nameText = document.createElement("div");
            nameText.style.display = "flex";
            nameText.style.alignItems = "center";
            nameText.style.color = "#fff";
            nameText.style.fontSize = "13px";
            
            // 1、2、3...序号
            const serialNum = document.createElement("span");
            serialNum.textContent = index + ". ";
            serialNum.style.color = "#88ccff";
            serialNum.style.marginRight = "8px";
            serialNum.style.fontWeight = "bold";

            // 完整文件名作为标题
            const fileName = document.createElement("span");
            fileName.textContent = file;

            nameText.appendChild(serialNum);
            nameText.appendChild(fileName);

            // 删除按钮（移除预览按钮）
            const btnGroup = document.createElement("div");
            btnGroup.style.display = "flex";
            btnGroup.style.gap = "6px";

            const delBtn = document.createElement("button");
            delBtn.textContent = "删除";
            delBtn.style.fontSize = "12px";
            delBtn.style.padding = "2px 6px";
            delBtn.style.backgroundColor = "#c72c2c";
            delBtn.style.color = "#fff";
            delBtn.style.border = "none";
            delBtn.style.borderRadius = "3px";
            delBtn.onclick = async function() {
                if (!confirm("确定删除 " + file + "？")) return;
                const isSuccess = await deletePrompt(subdirWidget.value, file);
                if (isSuccess) {
                    await updateList();
                }
            };

            btnGroup.appendChild(delBtn);
            item.appendChild(nameText);
            item.appendChild(btnGroup);
            listWrap.appendChild(item);
        });

        // 空列表提示
        if (files.length === 0) {
            const emptyTip = document.createElement("div");
            emptyTip.style.padding = "12px";
            emptyTip.style.textAlign = "center";
            emptyTip.style.color = "#888";
            emptyTip.textContent = "暂无提示词文件";
            listWrap.appendChild(emptyTip);
        }
    }

    // 目录切换自动刷新列表
    const oldCb = subdirWidget.callback;
    subdirWidget.callback = function(v) {
        if (oldCb) oldCb.call(this, v);
        updateList();
    };

    // 初始加载列表
    updateList();
}

app.registerExtension({
    name: "FxPromptManager",
    async nodeCreated(node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(function() { addUI(node); }, 100);
        }
},
});