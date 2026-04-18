import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "FxAiPromptManager";

// 获取提示词文件列表
async function fetchFileList(subdir) {
    const resp = await fetch(api.apiURL(`/fxpromptmanager/list?subdir=${encodeURIComponent(subdir)}`));
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.files;
}

// 上传提示词文件（支持自定义文件名）
async function uploadFiles(files, subdir, customFilename, onProgress) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        let finalName = "";

        // 有自定义文件名就用，没有就用原文件名
        if (customFilename && customFilename.trim() !== "") {
            const ext = file.name.split('.').pop().toLowerCase();
            finalName = customFilename.trim() + "." + ext;
        } else {
            finalName = file.name;
        }

        // 过滤非法字符
        finalName = finalName.replace(/[\\/*?:"<>|]/g, "");
        if (!finalName) finalName = "prompt.txt";

        const formData = new FormData();
        formData.append("prompt", file, finalName);
        formData.append("subdir", subdir);

        try {
            const response = await fetch(api.apiURL("/fxpromptmanager/upload"), {
                method: "POST",
                body: formData,
            });
            if (!response.ok) throw new Error(`上传失败 ${response.status}`);
            onProgress?.(i, 1);
        } catch (err) {
            throw new Error(`${finalName} 上传失败：${err.message}`);
        }
    }
}

// 预览提示词内容
async function previewPrompt(subdir, filename) {
    try {
        const resp = await fetch(api.apiURL(`/fxpromptmanager/preview?subdir=${encodeURIComponent(subdir)}&filename=${encodeURIComponent(filename)}`));
        if (!resp.ok) throw new Error("预览失败");
        const content = await resp.text();
        alert(`【${filename}】\n\n${content}`);
    } catch (err) {
        alert("预览失败：" + err.message);
    }
}

// 阻止默认拖拽打开
function preventDefaultDragDrop() {
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());
    document.addEventListener('dragenter', e => e.preventDefault());
    document.addEventListener('dragleave', e => e.preventDefault());
}

// 构建 UI
function addUI(node) {
    if (node._uiAdded) return;
    node._uiAdded = true;
    preventDefaultDragDrop();

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
    // 【自定义文件名输入框】
    // ======================
    const nameWrap = document.createElement("div");
    nameWrap.style.display = "flex";
    nameWrap.style.gap = "8px";
    nameWrap.style.marginBottom = "8px";
    nameWrap.style.alignItems = "center";

    const label = document.createElement("div");
    label.textContent = "文件名：";
    label.style.color = "#ccc";

    const filenameInput = document.createElement("input");
    filenameInput.type = "text";
    filenameInput.placeholder = "留空则用原文件名（不用输 .txt）";
    filenameInput.style.flex = 1;
    filenameInput.style.padding = "5px";
    filenameInput.style.backgroundColor = "#2a2a2a";
    filenameInput.style.color = "#fff";
    filenameInput.style.border = "1px solid #666";
    filenameInput.style.borderRadius = "4px";

    nameWrap.appendChild(label);
    nameWrap.appendChild(filenameInput);
    container.appendChild(nameWrap);

    // 拖拽上传区
    const dropArea = document.createElement("div");
    dropArea.style.padding = "12px";
    dropArea.style.border = "2px dashed #777";
    dropArea.style.borderRadius = "6px";
    dropArea.style.textAlign = "center";
    dropArea.style.color = "#ccc";
    dropArea.textContent = "📥 拖拽 .txt 提示词文件到这里上传";
    container.appendChild(dropArea);

    dropArea.addEventListener("dragover", e => {
        e.preventDefault();
        dropArea.style.borderColor = "#fff";
    });
    dropArea.addEventListener("dragleave", () => {
        dropArea.style.borderColor = "#777";
    });
    dropArea.addEventListener("drop", async e => {
        e.preventDefault();
    e.stopPropagation();
    dropArea.style.borderColor = "#777";
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.txt'));
    if (!files.length) { alert("只支持 .txt 文件"); return; }

    uploadBtn.disabled = true;
    const oldText = uploadBtn.textContent;
    uploadBtn.textContent = "上传中...";
    try {
        await uploadFiles(files, subdirWidget.value, filenameInput.value, () => {});
        filenameInput.value = "";
        await updateList();
    } catch (e) {
        alert(e.message);
    } finally {
        uploadBtn.textContent = oldText;
        uploadBtn.disabled = false;
    }
});

// 按钮组
const btnWrap = document.createElement("div");
btnWrap.style.display = "flex";
btnWrap.style.gap = "8px";
btnWrap.style.marginTop = "8px";
btnWrap.style.marginBottom = "8px";

const uploadBtn = document.createElement("button");
uploadBtn.textContent = "📤 选择文件上传";

const refreshBtn = document.createElement("button");
refreshBtn.textContent = "🔄 刷新列表";

btnWrap.appendChild(uploadBtn);
btnWrap.appendChild(refreshBtn);
container.appendChild(btnWrap);

// 文件列表
const listWrap = document.createElement("div");
listWrap.style.maxHeight = "400px";
listWrap.style.overflowY = "auto";
listWrap.style.border = "1px solid #666";
listWrap.style.padding = "4px";
container.appendChild(listWrap);

// 点击上传
uploadBtn.onclick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".txt";
    input.onchange = async () => {
        const files = Array.from(input.files).filter(f => f.name.endsWith('.txt'));
        if (!files.length) return;

        uploadBtn.disabled = true;
        const oldText = uploadBtn.textContent;
        uploadBtn.textContent = "上传中...";
        try {
            await uploadFiles(files, subdirWidget.value, filenameInput.value, () => {});
            filenameInput.value = "";
            await updateList();
        } catch (e) {
            alert(e.message);
        } finally {
            uploadBtn.textContent = oldText;
            uploadBtn.disabled = false;
        }
    };
    input.click();
};

// 刷新
refreshBtn.onclick = updateList;

// 更新列表（无排序、无拖拽）
async function updateList() {
    const files = await fetchFileList(subdirWidget.value);
    listWrap.innerHTML = "";

    files.forEach(file => {
        const item = document.createElement("div");
        item.style.padding = "6px 8px";
        item.style.backgroundColor = "#222";
        item.style.borderRadius = "4px";
        item.style.marginBottom = "4px";
        item.style.display = "flex";
        item.style.justifyContent = "space-between";
        item.style.alignItems = "center";

        const nameText = document.createElement("div");
        nameText.textContent = file;
        nameText.style.color = "#fff";
        nameText.style.fontSize = "13px";

        const btnGroup = document.createElement("div");
        btnGroup.style.display = "flex";
        btnGroup.style.gap = "6px";

        const previewBtn = document.createElement("button");
        previewBtn.textContent = "预览";
        previewBtn.style.fontSize = "12px";
        previewBtn.style.padding = "2px 6px";
        previewBtn.style.backgroundColor = "#444";
        previewBtn.style.color = "#fff";
        previewBtn.style.border = "none";
        previewBtn.style.borderRadius = "3px";
        previewBtn.onclick = () => previewPrompt(subdirWidget.value, file);

        const delBtn = document.createElement("button");
        delBtn.textContent = "删除";
        delBtn.style.fontSize = "12px";
        delBtn.style.padding = "2px 6px";
        delBtn.style.backgroundColor = "#c72c2c";
        delBtn.style.color = "#fff";
        delBtn.style.border = "none";
        delBtn.style.borderRadius = "3px";
        delBtn.onclick = async () => {
            if (!confirm(`确定删除 ${file}？`)) return;
            // 前端删除列表项，后端下次刷新会同步
            item.remove();
        };

        btnGroup.appendChild(previewBtn);
        btnGroup.appendChild(delBtn);
        item.appendChild(nameText);
        item.appendChild(btnGroup);
        listWrap.appendChild(item);
    });
}

// 目录切换刷新
const oldCb = subdirWidget.callback;
subdirWidget.callback = (v) => {
    oldCb?.call(this, v);
    updateList();
};

updateList();
}

app.registerExtension({
    name: "FxPromptManager",
    async nodeCreated(node) {
        if (node.comfyClass === TARGET_CLASS) {
            setTimeout(() => addUI(node), 100);
        }
},
});