import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ==============================================
// 核心：获取当前目录文件列表
// ==============================================
function fetchFileList(subdir) {
    return new Promise(resolve => {
        const url = api.apiURL("/fxai/image/v2/list?subdir=" + encodeURIComponent(subdir));
        fetch(url)
            .then(resp => resp.ok ? resp.json() : { files: [] })
            .then(data => resolve(data.files || []))
            .catch(() => resolve([]));
    });
}

// ==============================================
// 图片选择器（纯索引序号）
// ==============================================
window.FxAiImageAssetsSelector = function(selectStr) {
    return new Promise(resolve => {
        let selectedIndexes = [];
        let currentFileList = [];
        let currentSubdir = "sucai";

        // 遮罩
        const mask = document.createElement("div");
        mask.style.cssText = `
            position: fixed; inset: 0; z-index: 99999;
            background: rgba(0,0,0,0.85);
            display: flex; align-items: center; justify-content: center;
        `;
        document.body.appendChild(mask);

        // 弹窗
        const modal = document.createElement("div");
        modal.style.cssText = `
            width: 900px; max-width: 95vw;
            height: 750px; max-height: 90vh;
            background: #222; border-radius: 10px;
            padding: 20px; box-sizing: border-box;
            display: flex; flex-direction: column; gap: 16px;
        `;
        mask.appendChild(modal);

        // 标题 + 目录输入框
        const header = document.createElement("div");
        header.style.cssText = "display: flex; justify-content: space-between; align-items: center;";
        modal.appendChild(header);

        const title = document.createElement("div");
        title.textContent = "🖼️ 图片选择器（纯索引序号）";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        header.appendChild(title);

        // 修复：目录输入框可正常输入
        const dirInput = document.createElement("input");
        dirInput.value = "sucai";
        dirInput.placeholder = "输入图片目录，默认 sucai";
        dirInput.style.cssText = `
            padding: 6px 12px; border-radius: 4px; background: #333; color: #fff;
            border: 1px solid #444; min-width: 200px; outline: none;
        `;
        header.appendChild(dirInput);

        // 切换目录
        const reloadDir = () => {
            currentSubdir = dirInput.value.trim();
            selectedIndexes = [];
            renderList();
            renderSelectedBar();
        };
        dirInput.onblur = reloadDir;
        dirInput.onkeydown = (e) => e.key === "Enter" && reloadDir();

        // 图片列表容器
        const listContainer = document.createElement("div");
        listContainer.style.cssText = `
            flex: 1; overflow-y: auto;
            display: flex; flex-wrap: wrap; gap: 6px;
            padding:4px; align-content:flex-start;
        `;
        modal.appendChild(listContainer);

        // 已选预览区
        const selectedWrap = document.createElement("div");
        selectedWrap.style.cssText = `
            min-height:86px; max-height:120px; overflow-x:auto; overflow-y:hidden;
            background:#2b2b2b; border-radius:6px; padding:10px;
            display:flex; gap:8px; align-items:center;
        `;
        modal.appendChild(selectedWrap);

        // 底部按钮
        const bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display: flex; justify-content: flex-end; gap: 10px;";
        modal.appendChild(bottomBar);

        const btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        btnCancel.style.cssText = `padding: 6px 16px; border: none; border-radius: 4px; background: #555; color: #fff; cursor: pointer;`;
        
        const btnConfirm = document.createElement("button");
        btnConfirm.textContent = "✅ 确认选择";
        btnConfirm.style.cssText = `padding: 6px 16px; border: none; border-radius: 4px; background: #4a8fff; color: #fff; cursor: pointer;`;
        
        bottomBar.append(btnCancel, btnConfirm);

        // 关闭窗口
        function close() { document.body.removeChild(mask); }
        btnCancel.onclick = () => { resolve(); close(); };
        
        // 确认返回序号
        btnConfirm.onclick = () => { 
            resolve(selectedIndexes.join(",")); 
            close(); 
        };
        
        mask.onclick = (e) => e.target === mask && close();

        // ==============================================
        // 渲染已选列表（只显示图片）
        // ==============================================
        function renderSelectedBar() {
            selectedWrap.innerHTML = "";
            if (selectedIndexes.length === 0) {
                selectedWrap.innerHTML = '<span style="color:#999;">暂无选中，点击图片添加</span>';
                return;
            }

            selectedIndexes.forEach((idx, order) => {
                const file = currentFileList[idx];
                if (!file) return;
                const filename = file.filename || file;

                const item = document.createElement("div");
                item.style.cssText = `
                    width:70px; height:70px; position:relative;
                    border:2px solid #4a8fff; border-radius:4px;
                    flex-shrink:0; background:#111; overflow:hidden;
                `;

                const img = document.createElement("img");
                img.src = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(currentSubdir)}&filename=${encodeURIComponent(filename)}`);
                img.style.cssText = "width:100%;height:100%;object-fit:cover;";

                const del = document.createElement("div");
                del.textContent = "×";
                del.style.cssText = `
                    position:absolute; top:0; right:0; width:18px; height:18px;
                    background:#f54242; color:#fff; font-size:14px; cursor:pointer;
                    text-align:center; line-height:18px;
                `;
                del.onclick = () => {
                    selectedIndexes = selectedIndexes.filter(i => i !== idx);
                    renderSelectedBar();
                    refreshImageSelectionUI(); // 只刷新选中状态，不重新加载列表
                };

                item.append(img, del);
                selectedWrap.appendChild(item);
            });
        }

        // ==============================================
        // 优化：只刷新选中样式，不重新请求图片
        // ==============================================
        function refreshImageSelectionUI() {
            const items = listContainer.querySelectorAll(".img-item");
            items.forEach((el, idx) => {
                const isSelected = selectedIndexes.includes(idx.toString());
                el.style.border = isSelected ? "3px solid #4a8fff" : "3px solid transparent";
                
                let numLayer = el.querySelector(".select-num");
                if (isSelected) {
                    const selOrder = selectedIndexes.indexOf(idx.toString()) + 1;
                    if (!numLayer) {
                        numLayer = document.createElement("div");
                        numLayer.className = "select-num";
                        numLayer.style.cssText = `
                            position:absolute; top:6px; left:6px; width:26px; height:26px;
                            background:#4a8fff; color:white; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-weight:bold; font-size:14px;
                        `;
                        el.appendChild(numLayer);
                    }
                    numLayer.textContent = selOrder;
                    numLayer.style.display = "flex";
                } else {
                    if (numLayer) numLayer.style.display = "none";
                }
            });
        }

        // ==============================================
        // 渲染图片列表（只加载一次）
        // ==============================================
        function renderList() {
            listContainer.innerHTML = "";
            fetchFileList(currentSubdir).then(files => {
                currentFileList = files;
                
                files.forEach((file, index) => {
                    const filename = file.filename || file;
                    const imgIndex = index.toString();

                    const item = document.createElement("div");
                    item.className = "img-item";
                    item.style.cssText = `
                        width:128px; height:128px; position:relative; border-radius:6px;
                        overflow:hidden; cursor:pointer; border:3px solid transparent;
                    `;

                    const img = document.createElement("img");
                    img.src = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(currentSubdir)}&filename=${encodeURIComponent(filename)}`);
                    img.style.cssText = "width:100%; height:100%; object-fit:cover;";

                    const tag = document.createElement("div");
                    tag.textContent = `索引：${imgIndex}`;
                    tag.style.cssText = `
                        position:absolute; bottom:0; left:0; padding:2px 8px;
                        background:rgba(0,0,0,0.7); color:#fff; font-size:12px;
                    `;

                    item.append(img, tag);
                    listContainer.appendChild(item);

                    // 点击切换选中
                    item.onclick = () => {
                        if (selectedIndexes.includes(imgIndex)) {
                            selectedIndexes = selectedIndexes.filter(i => i !== imgIndex);
                        } else {
                            selectedIndexes.push(imgIndex);
                        }
                        renderSelectedBar();
                        refreshImageSelectionUI(); // 只刷新UI，不重新加载图片
                    };
                });

                refreshImageSelectionUI();
            });
        }

        // 初始化
        renderList();
        renderSelectedBar();
    });
};

// 注册插件
app.registerExtension({
    name: "FxAiImageSelectorIndex"
});