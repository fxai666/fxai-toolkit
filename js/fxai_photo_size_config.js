import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FxAiPhotoSizeConfigV2",
    nodeCreated: function(node) {
        if (node.comfyClass !== "FxAiPhotoSizeConfigV2") {
            return;
        }

        var presetCombo = null;
        var inputW = null;
        var inputH = null;

        // 遍历所有控件
        for (var i = 0; i < node.widgets.length; i++) {
            var widget = node.widgets[i];
            switch (widget.name) {
                case "预设":
                    presetCombo = widget;
                    break;
                case "宽度":
                    inputW = widget;
                    break;
                case "高度":
                    inputH = widget;
                    break;
            }
        }
        if (!presetCombo || !inputW || !inputH) {
            return;
        }

        // 解析预设宽高
        function parseSize(text) {
            var parts = text.split(" ");
            var sizeStr = parts[parts.length - 1];
            var arr = sizeStr.split("×");
            if (arr.length !== 2) {
                return null;
            }
            var w = parseInt(arr[0].trim(), 10);
            var h = parseInt(arr[1].trim(), 10);
            if (isNaN(w) || isNaN(h)) {
                return null;
            }
            return { w: w, h: h };
        }

        // 下拉切换回调
        presetCombo.callback = function() {
            var size = parseSize(presetCombo.value);
            if (!size) {
                return;
            }
            inputW.value = size.w;
            inputH.value = size.h;
            //node.setDirty(true);
        };

        // 初始化赋值
        var initSize = parseSize(presetCombo.value);
        if (initSize) {
            inputW.value = initSize.w;
            inputH.value = initSize.h;
        }
    }
});