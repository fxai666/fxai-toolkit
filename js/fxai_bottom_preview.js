import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "fxai-bottom-preview",
    nodeCreated: function(node) {
        if (document.getElementById('fxai-bottom-preview')) return;

        // 底部预览栏
        var previewEl = document.createElement('div');
        previewEl.id = 'fxai-bottom-preview';
        previewEl.dataset.expanded = "true";
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
            "flex-wrap: wrap;" + 
            "gap: 8px;" +
            "padding: 8px;" +
            "overflow: auto;" +
            "z-index: 1;" +
            "box-sizing: border-box;" +
            "transition: all 0.3s ease;";

        // 左侧缩放按钮
        const toggleBtn = document.createElement('button');
        toggleBtn.innerText = '❮';
        toggleBtn.style.cssText =
            "position:fixed;left:65px;bottom:5px;" +
            "width:24px;height:24px;border-radius:50%;border:none;background:#444;color:#fff;" +
            "cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:1;";
        toggleBtn.onclick = () => {
            const isOpen = previewEl.dataset.expanded === "true";
            if(isOpen){
                // 收缩成左下角小方块
                previewEl.dataset.expanded = "false";
                previewEl.style.right = "auto";
                previewEl.style.width = "35px";
                previewEl.style.transform ="all 0.3s ease";
                previewEl.style.height = "35px";
                previewEl.style.overflow = "hidden";
                toggleBtn.innerText = '❯';
            }else{
                toggleBtn.innerText = '❮';
                // 展开回底部长条
                previewEl.dataset.expanded = "true";
                previewEl.style.width = "auto";
                previewEl.style.transform ="all 0.3s ease";
                previewEl.style.right = "5px";
                previewEl.style.overflow = "auto";
                previewEl.style.height = "120px";
            }
        };
        previewEl.appendChild(toggleBtn);
        document.body.appendChild(previewEl);

        var style = document.createElement('style');
        style.textContent = ".litegraph { padding-bottom: 130px !important; }";
        document.head.appendChild(style);

        // 全局弹窗查看器 + 左右箭头切换功能
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

            let scale = 1;
            let translateX = 0;
            let translateY = 0;
            let isDragging = false;
            let dragStartX = 0;
            let dragStartY = 0;
            
            // 新增：键盘切换变量
            let currentKeydownListener = null;

            const closeViewer = () => {
                viewer.style.display = "none";
                scale = 1;
                translateX = 0;
                translateY = 0;
                bigImg.style.transform = `scale(${scale}) translate(${translateX}px,${translateY}px)`;
                
                // 关闭时移除键盘事件
                if (currentKeydownListener) {
                    document.removeEventListener("keydown", currentKeydownListener);
                    currentKeydownListener = null;
                }
            };
            closeBtn.onclick = closeViewer;
            viewer.onclick = (e) => { if(e.target === viewer) closeViewer(); };

            viewer.onwheel = (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                scale = Math.max(0.2, Math.min(5, scale + delta));
                bigImg.style.transform = `scale(${scale}) translate(${translateX}px,${translateY}px)`;
            };

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

            // 打开图片 + 绑定键盘左右箭头
            window.fxaiOpenImage = (src) => {
                bigImg.src = src;
                viewer.style.display = "flex";
                
                // 先清除旧事件
                if (currentKeydownListener) {
                    document.removeEventListener("keydown", currentKeydownListener);
                }
                
                // 新键盘事件
                currentKeydownListener = (e) => {
                    if (viewer.style.display !== "flex") return;
                    
                    const images = Array.from(previewEl.querySelectorAll("img"));
                    if (images.length === 0) return;
                    
                    let currentIndex = images.findIndex(img => img.src === bigImg.src);
                    if (currentIndex === -1) return;

                    // 左箭头 ← 上一张
                    if (e.key === "ArrowLeft") {
                        currentIndex = (currentIndex - 1 + images.length) % images.length;
                    }
                        // 右箭头 → 下一张
                    else if (e.key === "ArrowRight") {
                        currentIndex = (currentIndex + 1) % images.length;
                    } else {
                        return;
                    }
                    
                    // 切换图片 & 重置缩放
                    bigImg.src = images[currentIndex].src;
                    scale = 1;
                    translateX = 0;
                    translateY = 0;
                    bigImg.style.transform = `scale(1) translate(0,0)`;
                };
                
                document.addEventListener("keydown", currentKeydownListener);
            };
        }

        // 监听生成图片
        api.addEventListener('executed', function(e) {
            var outputs = e.detail && e.detail.output;
            if (!outputs || !outputs["images"]) return;

            var images = outputs["images"];

            for (var index in images) {
                var imgInfo = images[index];
                var src = "/view?filename=" + encodeURIComponent(imgInfo.filename) +
                          "&subfolder=" + encodeURIComponent(imgInfo.subfolder) +
                          "&type=" + imgInfo.type +
                          "&_t=" + new Date().getTime();

                var imgWrapper = document.createElement('div');
                imgWrapper.style.position = 'relative';
                imgWrapper.style.display = 'inline-block';
                imgWrapper.style.borderRadius = '10px';

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