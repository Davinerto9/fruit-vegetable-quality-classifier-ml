"""
Segmenta la fruta de una foto, decide si la foto sirve y genera los recortes para los modelos.

Fase CRISP-DM: Preparación de datos (también se reutiliza en Despliegue, dentro de la app)
Uso: python -m src.data.segmentation <ruta_imagen>

Pasos de process():
1. Reduce la imagen a config.SEG_WORK_SIZE y la pasa a Lab.
2. Descarta fotos en escala de grises.
3. Estima los colores del fondo con k-means sobre el borde de la imagen y marca como fruta
   los píxeles lejos de todos ellos (máscara inicial, umbral de Otsu).
4. Ajusta el borde con GrabCut y se queda con el objeto principal, con los huecos rellenos.
5. Cuenta las frutas (objetos separados + frutas pegadas, separadas con watershed) y revisa
   que la máscara sea razonable.
6. Entrega dos recortes cuadrados con el mismo encuadre: con fondo (CNN) y sin fondo (SVM/RF).
"""
import sys

import cv2
import numpy as np

from src import config


def to_work_size(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Reduce la imagen para segmentar más rápido y calcula su versión Lab suavizada.

    Parámetros:
        img: imagen BGR (H, W, 3) uint8.

    Retorna:
        (imagen reducida BGR, imagen Lab float32 de la reducida, factor de escala aplicado).
    """
    h, w = img.shape[:2]
    scale = config.SEG_WORK_SIZE / max(h, w)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return small, lab, scale


def is_grayscale(lab: np.ndarray) -> bool:
    """
    Indica si la foto casi no tiene color (por ejemplo, una foto en blanco y negro).

    Parámetros:
        lab: imagen Lab float32 (H, W, 3) de OpenCV, donde a = b = 128 es gris.

    Retorna:
        True si el promedio de |a - 128| + |b - 128| es menor que config.GRAY_TOL.
    """
    chroma = np.abs(lab[:, :, 1] - 128) + np.abs(lab[:, :, 2] - 128)
    return float(np.mean(chroma)) < config.GRAY_TOL


def background_colors(lab: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Estima los colores típicos del fondo con k-means sobre un borde del 6 % de la imagen.

    Parámetros:
        lab: imagen Lab float32 (H, W, 3).
        k: número máximo de colores de fondo (un piso puede tener dos tonos).

    Retorna:
        Centros Lab (m, 3) con m <= k; se descartan los grupos con menos del 15 % del borde
        (por ejemplo, la punta de la fruta que toca el borde).

    Fuente: k-means++ (Arthur y Vassilvitskii, 2007), implementación cv2.kmeans de OpenCV.
    """
    b = max(2, int(0.06 * min(lab.shape[:2])))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(config.SEED)
    _, labels, centers = cv2.kmeans(border, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    return centers[counts > 0.15 * len(border)]


def initial_mask(lab: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Máscara inicial: píxeles cuyo color está lejos de todos los colores de fondo.

    Parámetros:
        lab: imagen Lab float32 (H, W, 3).
        centers: colores de fondo (m, 3) de background_colors().

    Retorna:
        Máscara binaria uint8 (H, W) con 255 = fruta.

    Fórmula: score = min_j ||(a, b) - (a_j, b_j)|| + max(0, min_j L_j - L - 60).
        La distancia usa solo cromaticidad (a, b) para que las sombras, que cambian el brillo
        pero no el color, no parezcan fruta; el segundo término recupera las manchas muy
        oscuras (podredumbre), que casi no tienen color. El umbral sale del método de Otsu.
    Fuente: N. Otsu, "A threshold selection method from gray-level histograms", 1979,
        https://doi.org/10.1109/TSMC.1979.4310076
    """
    dist = np.min([np.linalg.norm(lab[:, :, 1:] - c[1:], axis=2) for c in centers], axis=0)
    dark = np.clip(np.min(centers[:, 0]) - lab[:, :, 0] - 60, 0, None)
    score = cv2.normalize(dist + dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(th, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Rellena los huecos internos de la máscara (manchas de la fruta que no superaron el umbral).

    Parámetros:
        mask: máscara binaria uint8 (H, W).

    Retorna:
        Máscara uint8 (H, W) con cada contorno externo relleno.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def border_touches(x: int, y: int, w: int, h: int, shape: tuple) -> int:
    """
    Cuenta cuántos bordes de la imagen toca un rectángulo.

    Parámetros:
        x, y, w, h: rectángulo del objeto (de cv2.connectedComponentsWithStats).
        shape: forma (alto, ancho) de la imagen.

    Retorna:
        Número de bordes tocados (0 a 4).
    """
    return int(x == 0) + int(y == 0) + int(x + w == shape[1]) + int(y + h == shape[0])


def main_object(mask: np.ndarray) -> np.ndarray:
    """
    Elige el objeto principal: el de mayor área, penalizando (x 0.25) los que tocan 2 o más bordes,
    porque suelen ser fondo (una mesa, otra fruta cortada en el borde).

    Parámetros:
        mask: máscara binaria uint8 (H, W).

    Retorna:
        Máscara uint8 (H, W) solo con el objeto principal, relleno; vacía si no hay objetos.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    scores = [stats[i][4] * (0.25 if border_touches(*stats[i][:4], mask.shape) >= 2 else 1.0)
              for i in range(1, n)]
    best = 1 + int(np.argmax(scores))
    return fill_holes(np.where(labels == best, 255, 0).astype(np.uint8))


def large_objects(mask: np.ndarray) -> np.ndarray:
    """
    Objeto principal más los demás objetos grandes (> 25 % de su área) que no tocan 2 bordes.
    Sirve para detectar fotos con varias frutas separadas.

    Parámetros:
        mask: máscara binaria uint8 (H, W).

    Retorna:
        Máscara uint8 (H, W) con todos esos objetos, rellenos.
    """
    out = main_object(mask)
    ref = cv2.countNonZero(out)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        if stats[i][4] > 0.25 * ref and border_touches(*stats[i][:4], mask.shape) < 2:
            out[labels == i] = 255
    return fill_holes(out)


def refine_grabcut(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Ajusta el borde de la máscara con GrabCut y se queda con el objeto principal.

    Parámetros:
        img: imagen BGR reducida (H, W, 3).
        mask: máscara inicial uint8 (H, W) del objeto principal.

    Retorna:
        Máscara refinada uint8 (H, W).

    Inicialización: núcleo erosionado = fruta segura; máscara = probable fruta;
        anillo dilatado = probable fondo; resto = fondo seguro.
    Fuente: C. Rother, V. Kolmogorov y A. Blake, "GrabCut: interactive foreground extraction
        using iterated graph cuts", 2004, https://doi.org/10.1145/1015706.1015720
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gc = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(mask, k, iterations=3) > 0] = cv2.GC_PR_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(mask, k, iterations=2) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return main_object(cv2.morphologyEx(out, cv2.MORPH_OPEN, k))


def distance_peaks(mask: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Centros candidatos de fruta: máximos locales de la transformada de distancia.

    Parámetros:
        mask: máscara binaria uint8 (H, W).

    Retorna:
        (transformada de distancia float32 (H, W), lista de puntos (x, y) de los picos
        con distancia > 35 % del máximo).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    k = max(7, int(0.08 * min(mask.shape)) | 1)
    peaks = (dist == cv2.dilate(dist, np.ones((k, k), np.uint8))) & (dist > 0.35 * dist.max())
    _, _, _, cents = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    return dist, [(int(cx), int(cy)) for cx, cy in cents[1:]]


def group_peaks(dist: np.ndarray, pts: list) -> list:
    """
    Agrupa los picos que pertenecen a la misma fruta.

    Parámetros:
        dist: transformada de distancia (H, W).
        pts: picos (x, y) de distance_peaks().

    Retorna:
        Lista con el grupo (entero) de cada pico.

    Criterio: dos picos son frutas distintas solo si la línea que los une pasa por un "cuello"
        donde la distancia baja de 75 % del pico menor; así un banano alargado no se parte.
    """
    group = list(range(len(pts)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            xs = np.linspace(pts[i][0], pts[j][0], 50).astype(int)
            ys = np.linspace(pts[i][1], pts[j][1], 50).astype(int)
            low = 0.75 * min(dist[pts[i][1], pts[i][0]], dist[pts[j][1], pts[j][0]])
            if dist[ys, xs].min() >= low:
                group[root(j)] = root(i)
    return [root(i) for i in range(len(pts))]


def count_touching(img: np.ndarray, mask: np.ndarray) -> int:
    """
    Cuenta las frutas pegadas dentro de una máscara separándolas con watershed.

    Parámetros:
        img: imagen BGR reducida (H, W, 3).
        mask: máscara uint8 (H, W) de un objeto.

    Retorna:
        Número de frutas con área > 25 % de la mayor.

    Fuente: tutorial de watershed de OpenCV,
        https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    """
    dist, pts = distance_peaks(mask)
    groups = group_peaks(dist, pts)
    roots = sorted(set(groups))
    if len(roots) <= 1:
        return 1
    markers = np.zeros(mask.shape, np.int32)
    markers[cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0] = 1
    for (x, y), g in zip(pts, groups):
        cv2.circle(markers, (x, y), max(2, int(0.3 * dist[y, x])), 2 + roots.index(g), -1)
    cv2.watershed(img, markers)
    areas = [cv2.countNonZero(fill_holes(((markers == 2 + i) & (mask > 0)).astype(np.uint8) * 255))
             for i in range(len(roots))]
    return sum(a > 0.25 * max(areas) for a in areas if a > 0)


def count_fruits(img: np.ndarray, initial: np.ndarray, mask: np.ndarray) -> int:
    """
    Número de frutas en la foto: el mayor entre los objetos grandes separados de la máscara
    inicial y las frutas pegadas dentro de la máscara final.

    Parámetros:
        img: imagen BGR reducida (H, W, 3).
        initial: máscara inicial uint8 (H, W).
        mask: máscara final uint8 (H, W).

    Retorna:
        Número estimado de frutas (>= 1).
    """
    n_separate, _ = cv2.connectedComponents(large_objects(initial))
    return max(n_separate - 1, count_touching(img, mask))


def square_crop(img: np.ndarray, box: tuple, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """
    Recorte cuadrado alrededor de un rectángulo, con margen config.CROP_MARGIN por lado.

    Parámetros:
        img: imagen (H, W, 3).
        box: rectángulo (x, y, w, h) de la fruta.
        border: relleno si el recorte se sale de la imagen: cv2.BORDER_REPLICATE (con fondo)
            o cv2.BORDER_CONSTANT (negro, para el recorte sin fondo).

    Retorna:
        Recorte cuadrado (S, S, 3) con S = max(w, h) * (1 + 2 * margen).
    """
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * config.CROP_MARGIN))
    padded = cv2.copyMakeBorder(img, side, side, side, side, border, value=0)
    x0, y0 = x + w // 2 - side // 2 + side, y + h // 2 - side // 2 + side
    return padded[y0:y0 + side, x0:x0 + side]


def center_crop(img: np.ndarray) -> np.ndarray:
    """
    Recorte de respaldo cuando la segmentación no es confiable: el cuadrado central más grande.

    Parámetros:
        img: imagen (H, W, 3).

    Retorna:
        Recorte cuadrado (min(H, W), min(H, W), 3).
    """
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def normalized_diameter(contour: np.ndarray, shape: tuple) -> float:
    """
    Diámetro equivalente de la fruta normalizado por el lado mayor de la imagen.

    Parámetros:
        contour: contorno de la fruta en coordenadas de la imagen original.
        shape: forma (alto, ancho) de la imagen original.

    Retorna:
        d_norm en [0, ~1].

    Fórmula: d = 2 * sqrt(A / pi), diámetro del círculo con la misma área A que la fruta;
        d_norm = d / max(alto, ancho).
    """
    return 2 * np.sqrt(cv2.contourArea(contour) / np.pi) / max(shape[:2])


def is_dubious(contour: np.ndarray, mask_small: np.ndarray, shape: tuple) -> bool:
    """
    Control de sanidad de la máscara.

    Parámetros:
        contour: contorno de la fruta en la imagen original.
        mask_small: máscara final en tamaño de trabajo.
        shape: forma (alto, ancho) de la imagen original.

    Retorna:
        True si la fruta ocupa menos de config.SEG_MIN_AREA o más de config.SEG_MAX_AREA de
        la imagen, o si la máscara toca 3 o 4 bordes (señal de que se tomó el fondo).
    """
    area = cv2.contourArea(contour) / (shape[0] * shape[1])
    touches = (int(mask_small[0].any()) + int(mask_small[-1].any())
               + int(mask_small[:, 0].any()) + int(mask_small[:, -1].any()))
    return area < config.SEG_MIN_AREA or area > config.SEG_MAX_AREA or touches >= 3


def process(img: np.ndarray) -> dict:
    """
    Procesa una foto completa: segmenta, revisa y recorta.

    Parámetros:
        img: imagen BGR (H, W, 3) uint8, como la entrega cv2.imread.

    Retorna:
        dict con:
          status: "ok", "discard: grayscale", "review: N fruits" o "review: dubious mask".
          mask: máscara uint8 (H, W) de la fruta (None si se descarta).
          contour: contorno de la fruta (None si se descarta).
          crop: recorte cuadrado CON fondo, para la CNN (recorte central si la máscara es dudosa).
          crop_nobg: mismo encuadre SIN fondo (negro), para las características del SVM/RF.
          d_norm: diámetro normalizado (None si se descarta).
    """
    small, lab, _ = to_work_size(img)
    if is_grayscale(lab):
        return {"status": "discard: grayscale", "mask": None, "contour": None,
                "crop": center_crop(img), "crop_nobg": None, "d_norm": None}
    initial = initial_mask(lab, background_colors(lab))
    mask_small = refine_grabcut(small, main_object(initial))
    mask = cv2.resize(mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "review: dubious mask", "mask": mask, "contour": None,
                "crop": center_crop(img), "crop_nobg": None, "d_norm": None}
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    box = cv2.boundingRect(contour)
    n_fruits = count_fruits(small, initial, mask_small)
    if is_dubious(contour, mask_small, img.shape):
        status, crop = "review: dubious mask", center_crop(img)
    else:
        status = "ok" if n_fruits == 1 else f"review: {n_fruits} fruits"
        crop = square_crop(img, box)
    return {"status": status, "mask": filled, "contour": contour, "crop": crop,
            "crop_nobg": square_crop(cv2.bitwise_and(img, img, mask=filled), box, cv2.BORDER_CONSTANT),
            "d_norm": normalized_diameter(contour, img.shape)}


if __name__ == "__main__":
    # Ejemplo mínimo: python -m src.data.segmentation data/own/banana_good_001.jpg
    result = process(cv2.imread(sys.argv[1]))
    d_norm = result["d_norm"]
    print(result["status"], "d_norm =", None if d_norm is None else round(d_norm, 3),
          "crop =", result["crop"].shape)
