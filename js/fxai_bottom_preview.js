import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "fxai-bottom-preview",
    nodeCreated: function(node) {
        if (document.getElementById('fxai-bottom-preview')) return;

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
        
        api.addEventListener('executed', function(e) {
            var outputs = e.detail && e.detail.output;
            if (!outputs || !outputs["images"]) return;

            var images = outputs["images"];
            console.log(images);

            for (var index in images) {
                var imgInfo = images[index];
                var src = "/view?filename=" + encodeURIComponent(imgInfo.filename) +
                          "&subfolder=" + encodeURIComponent(imgInfo.subfolder) +
                          "&type=" + imgInfo.type;

                // 创建包裹图片的容器
                var imgWrapper = document.createElement('div');
                imgWrapper.style.position = 'relative';
                imgWrapper.style.display = 'inline-block';
                imgWrapper.style.margin = '4px'; // 图片之间间距

                // 创建删除按钮
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
                    window.open(this.src, '_blank');
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