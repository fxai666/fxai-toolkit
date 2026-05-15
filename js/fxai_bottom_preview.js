import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "fxai-bottom-preview",
    nodeCreated: function(node) {
        if (document.getElementById('fxai-bottom-preview')) return;

        // ========== 底部预览栏 ==========
        var previewEl = document.createElement('div');
        previewEl.id = 'fxai-bottom-preview';
        previewEl.style.cssText =
            "position: fixed;" +
            "bottom: 0;" +
            "left: 60px;" +
            "right: 5px;" +
            "height: 120px;" +
            "background:#202020;" +
            "border-top: 1px solid #333;" +
            "border-radius:10px;" +
            "display: flex;" +
            "gap: 8px;" +
            "padding: 8px;" +
            "overflow-x: auto;" +
            "z-index: 1;" +
            "box-sizing: border-box;";
        document.body.appendChild(previewEl);

        var style = document.createElement('style');
        style.textContent = ".litegraph { padding-bottom: 130px !important; }";
        document.head.appendChild(style);

        // ========== 全局弹窗查看器（单例，只创建一次） ==========
        if (!document.getElementById("fxai-image-viewer")) {
            const viewer = document.createElement("div");
            viewer.id = "fxai-image-viewer";
            viewer.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0,0,0,0.92);
                z-index: 9999;
                display: none;
                align-items: center;
                justify-content: center;
                cursor: grab;
            `;

            const bigImg = document.createElement("img");
            bigImg.style.cssText = `
                max-width: 90vw;
                max-height: 90vh;
                object-fit: contain;
                pointer-events: none;
            `;

            const closeBtn = document.createElement("div");
            closeBtn.innerText = "×";
            closeBtn.style.cssText = `
                position: absolute;
                top: 20px;
                right: 30px;
                color: #fff;
                font-size: 36px;
                cursor: pointer;
                user-select: none;
            `;

            viewer.appendChild(bigImg);
            viewer.appendChild(closeBtn);
            document.body.appendChild(viewer);

            // 缩放&拖拽变量
            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let isDragging = false;
            let dragStartX = 0;
            let dragStartY = 0;

            // 关闭
            const closeViewer = () => {
                viewer.style.display = "none";
                scale = 1;
                translateX = 0;
                translateY = 0;
                bigImg.style.transform = `scale(${scale}) translate(${translateX}px,${translateY}px)`;
            };
            closeBtn.onclick = closeViewer;
            viewer.onclick = (e) => { if(e.target === viewer) closeViewer(); };

            // 滚轮缩放
            viewer.onwheel = (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                scale = Math.max(0.2, Math.min(5, scale + delta));
                bigImg.style.transform = `scale(${scale}) translate(${translateX}px,${translateY}px)`;
            };

            // 拖拽
            viewer.onmousedown = (e) => {
                isDragging = true;
                dragStartX = e.clientX - translateX;
                dragStartY = e.clientY - translateY;
                viewer.style.cursor = "grabbing";
            };
            document.onmousemove = (e) => {
                if(!isDragging) return;
                translateX = e.clientX - dragStartX;
                translateY = e.clientY - dragStartY;
                bigImg.style.transform = `scale(${scale}) translate(${translateX}px,${translateY}px)`;
            };
            document.onmouseup = () => {
                isDragging = false;
                viewer.style.cursor = "grab";
            };

            // 对外暴露打开方法
            window.fxaiOpenImage = (src) => {
                bigImg.src = src;
                viewer.style.display = "flex";
            };
        }

        // ========== 监听生成图片 ==========
        api.addEventListener('executed', function(e) {
            var outputs = e.detail && e.detail.output;
            if (!outputs || !outputs["images"]) return;

            var images = outputs["images"];

            for (var index in images) {
                var imgInfo = images[index];
                var src = "/view?filename=" + encodeURIComponent(imgInfo.filename) +
                          "&subfolder=" + encodeURIComponent(imgInfo.subfolder) +
                          "&type=" + imgInfo.type;

                var imgWrapper = document.createElement('div');
                imgWrapper.style.position = 'relative';
                imgWrapper.style.display = 'inline-block';
                imgWrapper.style.margin = '4px';

                var deleteBtn = document.createElement('button');
                deleteBtn.innerText = '×';
                deleteBtn.style.position = 'absolute';
                deleteBtn.style.top = '2px';
                deleteBtn.style.right = '2px';
                deleteBtn.style.background = '#ff4444';
                deleteBtn.style.color = '#fff';
                deleteBtn.style.border = 'none';
                deleteBtn.style.borderRadius = '50%';
                deleteBtn.style.width = '20px';
                deleteBtn.style.height = '20px';
                deleteBtn.style.cursor = 'pointer';
                deleteBtn.style.fontSize = '14px';
                deleteBtn.style.fontWeight = 'bold';
                deleteBtn.style.lineHeight = '20px';
                deleteBtn.style.padding = '0';
                deleteBtn.onclick = function() {
                    this.parentElement.remove();
                };

                var img = new Image();
                img.src = src;
                img.style.height = '100px';
                img.style.borderRadius = '4px';
                img.style.cursor = 'pointer';

                // 核心改动：调用弹窗查看器，不新开窗口
                img.onclick = function() {
                    window.fxaiOpenImage(this.src);
                };

                imgWrapper.appendChild(deleteBtn);
                imgWrapper.appendChild(img);

                if (previewEl.firstChild) {
                    previewEl.insertBefore(imgWrapper, previewEl.firstChild);
                } else {
                    previewEl.appendChild(imgWrapper);
                }
            }
        });
    }
});