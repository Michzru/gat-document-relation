import torch
from doclayout_yolo import YOLOv10
from doclayout_yolo.utils import ops

# --- ZAČIATOK OPRAVY (PATCH) ---
# Uložíme si pôvodnú funkciu, aby sme ju nezničili
original_nms = ops.non_max_suppression


def patched_nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(),
                max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, in_place=True, rotated=False):
    """
    Táto funkcia skontroluje, či model nevrátil slovník (dict).
    Ak áno, vytiahne z neho kľúčovú časť 'one2one', ktorú potrebuje NMS.
    """
    if isinstance(prediction, dict):
        # DocLayout/YOLOv10 zvykne vracať dict s kľúčmi 'one2one' a 'one2many'
        if 'one2one' in prediction:
            prediction = prediction['one2one']
        else:
            # Ak nepoznáme kľúč, vezmeme prvú hodnotu (záchranná brzda)
            prediction = list(prediction.values())[0]

    # Zavoláme pôvodnú funkciu s opravenými dátami
    return original_nms(prediction, conf_thres, iou_thres, classes, agnostic, multi_label, labels, max_det, nc,
                        max_time_img, max_nms, max_wh, in_place, rotated)


# Prepíšeme chybnú funkciu v knižnici našou opravenou verziou
ops.non_max_suppression = patched_nms
# --- KONIEC OPRAVY ---

# TERAZ UŽ TVOJ KÓD PÔJDE:
model = YOLOv10('doclayout_yolo_docstructbench_imgsz1024.pt')

results = model.predict(
    'data/PNG/fecf39d4d98045cc36c496707863aacb8214f60f527210b620c0eff708c4bbca.png',
    save=True,
    imgsz=1024,
    conf=0.25  # voliteľné: prah istoty
)