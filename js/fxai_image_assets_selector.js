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
        // 只存【图片索引数字】（第0张、第1张、第2张...）
        let selectedIndexes = [];
        // 当前目录文件列表（用于取索引）
        let currentFileList = [];
        // 默认目录：sucai
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

        // 标题 + 目录输入框（默认 sucai）
        const header = document.createElement("div");
        header.style.cssText = "display: flex; justify-content: space-between; align-items: center;";
        modal.appendChild(header);

        const title = document.createElement("div");
        title.textContent = "🖼️ 图片选择器（纯索引序号）";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        header.appendChild(title);

        // 目录手动输入框，默认 sucai
        const dirInput = document.createElement("input");
        dirInput.value = "sucai";
        dirInput.placeholder = "输入图片目录，默认 sucai";
        dirInput.style.cssText = `
            padding: 6px 12px; border-radius: 4px; background: #333; color: #fff;
            border: 1px solid #444; min-width: 200px; outline: none;
        `;
        header.appendChild(dirInput);

        // 切换目录 → 清空选中 + 刷新图片
        const reloadDir = () => {
            currentSubdir = dirInput.value.trim();
            selectedIndexes = []; // 切换目录 清空选中
            renderList();
            renderSelectedBar();
        };
        dirInput.onchange = reloadDir;
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
        
        // ✅ 确认：只返回 纯索引序号（逗号分隔）
        btnConfirm.onclick = () => { 
            resolve(selectedIndexes.join(",")); 
            close(); 
        };
        
        mask.onclick = (e) => e.target === mask && close();

        // ==============================================
        // 渲染已选序号
        // ==============================================
        function renderSelectedBar() {
            selectedWrap.innerHTML = "";
            if (selectedIndexes.length === 0) {
                selectedWrap.innerHTML = '<span style="color:#999;">暂无选中，点击图片添加</span>';
                return;
            }

            selectedIndexes.forEach((idx, order) => {
                const item = document.createElement("div");
                item.style.cssText = `
                    width:70px; height:70px; position:relative;
                    border:2px solid #4a8fff; border-radius:4px;
                    flex-shrink:0; background:#111;
                    display:flex; align-items:center; justify-content:center;
                    color:#fff; font-size:20px; font-weight:bold;
                `;
                item.textContent = idx;

                // 选中顺序编号
                const orderTag = document.createElement("div");
                orderTag.textContent = order + 1;
                orderTag.style.cssText = `
                    position:absolute; top:0; left:0; width:22px; height:22px;
                    background:#4a8fff; color:#fff; font-size:12px;
                    text-align:center; line-height:22px; border-radius:0 0 4px 0;
                `;

                // 删除按钮
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
                    renderList();
                };

                item.append(orderTag, del);
                selectedWrap.appendChild(item);
            });
        }

        // ==============================================
        // 渲染图片列表（核心：用数组索引当序号）
        // ==============================================
        function renderList() {
            listContainer.innerHTML = "";
            fetchFileList(currentSubdir).then(files => {
                currentFileList = files; // 保存当前文件列表
                
                files.forEach((file, index) => {
                    const filename = file.filename || file;
                    // ✅ 序号 = 数组索引（第0张、第1张、第2张...）
                    const imgIndex = index.toString();
                    const isSelected = selectedIndexes.includes(imgIndex);

                    const item = document.createElement("div");
                    item.style.cssText = `
                        width:128px; height:128px; position:relative; border-radius:6px;
                        overflow:hidden; cursor:pointer; border:3px solid ${isSelected ? "#4a8fff" : "transparent"};
                    `;

                    // 预览图
                    const img = document.createElement("img");
                    img.src = api.apiURL(`/fxai/image/v2/preview?subdir=${encodeURIComponent(currentSubdir)}&filename=${encodeURIComponent(filename)}`);
                    img.style.cssText = "width:100%; height:100%; object-fit:cover;";

                    // 显示：索引序号
                    const tag = document.createElement("div");
                    tag.textContent = `索引：${imgIndex}`;
                    tag.style.cssText = `
                        position:absolute; bottom:0; left:0; padding:2px 8px;
                        background:rgba(0,0,0,0.7); color:#fff; font-size:12px;
                    `;

                    item.append(img, tag);
                    listContainer.appendChild(item);

                    // 点击选中/取消
                    item.onclick = () => {
                        if (selectedIndexes.includes(imgIndex)) {
                            selectedIndexes = selectedIndexes.filter(i => i !== imgIndex);
                        } else {
                            selectedIndexes.push(imgIndex);
                        }
                        renderSelectedBar();
                        renderList();
                    };
                });
            });
        }

        // 初始化加载
        renderList();
        renderSelectedBar();
    });
};

// 注册插件
app.registerExtension({
    name: "FxAiImageSelectorIndex"
});