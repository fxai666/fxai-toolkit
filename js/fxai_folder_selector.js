import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * 目录选择弹窗 ES5
 * @param {string} initSubdir 初始打开子目录
 * @returns {Promise|null}
 */
window.FxAiFolderSelector = function (initSubdir) {
    return new Promise(function (resolve) {
        var currentPath = initSubdir ? initSubdir.trim() : "";
        var folderList = [];
        var selectedFolderName = null;
        var searchKeyword = "";

        // 遮罩
        var mask = document.createElement("div");
        mask.style.cssText = "position: fixed; inset: 0; z-index: 99999; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center;";
        document.body.appendChild(mask);

        // 弹窗
        var modal = document.createElement("div");
        modal.style.cssText = "width: 700px;height: 600px; background: #222; border-radius: 10px; padding: 20px; box-sizing: border-box; display: flex; flex-direction: column; gap: 16px;";
        mask.appendChild(modal);

        // 头部
        var header = document.createElement("div");
        header.style.cssText = "display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap:8px;";
        modal.appendChild(header);

        var title = document.createElement("div");
        title.textContent = "📁 凤希目录选择器";
        title.style.cssText = "font-size: 18px; color: #fff; font-weight: bold;";
        header.appendChild(title);

        var pathTip = document.createElement("div");
        pathTip.style.cssText = "color:#aaa; font-size:13px;";
        header.appendChild(pathTip);

        // 搜索栏容器
        var searchWrap = document.createElement("div");
        searchWrap.style.cssText = "width:100%; display:flex; gap:8px; align-items:center;";
        modal.appendChild(searchWrap);

        var searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = "🔍 搜索文件夹名称";
        searchInput.style.cssText = "flex:1; padding:8px 12px; background:#2b2b2b; border:1px solid #444; border-radius:4px; color:#fff; outline:none;";

        var searchClearBtn = document.createElement("button");
        searchClearBtn.textContent = "清空";
        searchClearBtn.style.cssText = "padding:6px 12px; border:none; border-radius:4px; background:#555; color:#fff; cursor:pointer;";
        searchWrap.appendChild(searchInput);
        searchWrap.appendChild(searchClearBtn);

        // 目录列表容器 修改为网格三列布局
        var folderWrap = document.createElement("div");
        folderWrap.style.cssText = "flex:1; overflow-y:auto; display: grid; grid-template-columns: repeat(3, 1fr); gap:8px; padding:4px;";
        modal.appendChild(folderWrap);

        // 底部按钮栏
        var bottomBar = document.createElement("div");
        bottomBar.style.cssText = "display:flex; justify-content:flex-end; gap:10px;";
        modal.appendChild(bottomBar);

        var btnCancel = document.createElement("button");
        btnCancel.textContent = "取消";
        btnCancel.style.cssText = "padding:6px 16px; border:none; border-radius:4px; background:#555; color:#fff; cursor:pointer;";

        var btnConfirm = document.createElement("button");
        btnConfirm.textContent = "确认选中";
        btnConfirm.style.cssText = "padding:6px 16px; border:none; border-radius:4px; background:#4a8fff; color:#fff; cursor:pointer;";
        bottomBar.appendChild(btnCancel);
        bottomBar.appendChild(btnConfirm);

        // 关闭弹窗
        function closeModal() {
            document.body.removeChild(mask);
        }

        btnCancel.onclick = function () {
            resolve();
            closeModal();
        };

        btnConfirm.onclick = function () {
            resolve(selectedFolderName);
            closeModal();
        };

        function loadFolderData(subdir) {
            var url = api.apiURL("/fxai/folder/list?subdir=" + encodeURIComponent(subdir));
            fetch(url).then(function (res) {
                if (!res.ok) {
                    return { sub_dirs: [] };
                }
                return res.json();
            }).then(function (data) {
                folderList = data.sub_dirs || [];
                currentPath = subdir;
                pathTip.textContent = "当前路径：fxai/" + currentPath;
                searchKeyword = searchInput.value.trim();
                renderFolderList();
            }).catch(function () {
                folderList = [];
                renderFolderList();
            });
        }

        // 过滤文件夹列表
        function getFilteredList() {
            if (!searchKeyword) return folderList;
            var kw = searchKeyword.toLowerCase();
            var result = [];
            for (var i = 0; i < folderList.length; i++) {
                var name = folderList[i].toLowerCase();
                if (name.indexOf(kw) !== -1) {
                    result.push(folderList[i]);
                }
            }
            return result;
        }

        // 渲染列表
        function renderFolderList() {
            folderWrap.innerHTML = "";
            var showList = getFilteredList();

            // 遍历过滤后的子目录
            for (var i = 0; i < showList.length; i++) {
                var name = showList[i];
                var item = document.createElement("div");
                item.className = "folder-item";
                // 调整item宽高适配网格
                item.style.cssText = "padding:10px 5px; background:#2b2b2b; border-radius:4px; color:#fff; cursor:pointer; display:flex; align-items:center; gap:8px; border:2px solid transparent; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;";
                item.innerHTML = "📂 " + name;

                // 单击选中
                item.onclick = function () {
                    var items = folderWrap.querySelectorAll(".folder-item");
                    for (var j = 0; j < items.length; j++) {
                        items[j].style.border = "2px solid transparent";
                    }
                    this.style.border = "2px solid #4a8fff";
                    selectedFolderName = this.innerText.replace("📂 ", "");
                };

                item.ondblclick = function () {
                    btnConfirm.onclick();
                };

                folderWrap.appendChild(item);
            }

            if (showList.length === 0) {
                var emptyTip = document.createElement("div");
                emptyTip.style.cssText = "color:#999; padding:20px; text-align:center; grid-column: 1 / -1;";
                emptyTip.textContent = searchKeyword ? "未匹配到相关文件夹" : "暂无子文件夹";
                folderWrap.appendChild(emptyTip);
            }
        }

        // 搜索输入实时过滤
        searchInput.oninput = function () {
            searchKeyword = this.value.trim();
            renderFolderList();
        };

        // 清空搜索框
        searchClearBtn.onclick = function () {
            searchInput.value = "";
            searchKeyword = "";
            renderFolderList();
        };

        // 初始加载
        loadFolderData(currentPath);
    });
};

app.registerExtension({
    name: "FxAiFolderSelector"
});