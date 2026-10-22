"""v2 rug models: calibrated albedo + fabric normal map + bound edge.
Albedo targets come from the rug as seen in real room footage / studio shots,
lifted slightly so AR lighting (which multiplies base color) lands near reality."""
import io, json, struct
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(11)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'src') + '/'
# key: (structure source, crop, real cm across crop, target base sRGB, detail amp)
SRC = {
    'green': (D + 'green_3141_6s.jpg', (60, 80, 1020, 1040), 35.0, (95, 137, 135), 0.10),
    'gray':  (D + 'gray_close.jpg', (1600, 850, 3100, 2350), 45.0, (178, 174, 171), 0.07),
    'beige': (D + 'beige_d3070.jpg', (1300, 250, 3050, 2000), 45.0, (204, 194, 184), 0.07),
    'ivory': (D + 'beige_d3070.jpg', (1300, 250, 3050, 2000), 45.0, (224, 218, 207), 0.06),
}
SIZES = [('S', 100, 150), ('M', 150, 200), ('L', 200, 250), ('XL', 200, 300),
         ('hall', 100, 250), ('widehall', 120, 200), ('module', 100, 200), ('kitchen', 45, 115)]
TH = 0.009  # 9 mm


def detail_src(key):
    f, box, cm, _, _ = SRC[key]
    im = Image.open(f).convert('L').crop(box).resize((960, 960), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    low = np.asarray(im.filter(ImageFilter.GaussianBlur(25))).astype(np.float32)
    hp = (a - low) / (low + 1)          # relative high-pass: fabric pile only, lighting removed
    hp = (hp - hp.mean()) / (hp.std() + 1e-6)
    return hp, cm


def splat(hp, cm, W, H, ppc):
    """random overlapping feathered patches -> seamless, non-repeating detail field"""
    scale = cm * ppc / hp.shape[0]
    s = np.asarray(Image.fromarray(hp.astype(np.float32), 'F').resize(
        (max(8, int(hp.shape[1] * scale)), max(8, int(hp.shape[0] * scale))), Image.BILINEAR))
    P = int(min(s.shape) * 0.7)
    yy, xx = np.mgrid[0:P, 0:P]
    mask = np.clip(1 - np.hypot(yy - P / 2, xx - P / 2) / (P / 2), 0, 1) ** 1.5
    acc = np.zeros((H, W), np.float32); ws = np.zeros((H, W), np.float32)
    step = max(4, int(P * 0.45))
    for y in range(-P // 2, H, step):
        for x in range(-P // 2, W, step):
            jx = x + rng.integers(-step // 3, step // 3 + 1); jy = y + rng.integers(-step // 3, step // 3 + 1)
            sy = rng.integers(0, s.shape[0] - P + 1); sx = rng.integers(0, s.shape[1] - P + 1)
            y0, x0, y1, x1 = max(jy, 0), max(jx, 0), min(jy + P, H), min(jx + P, W)
            if y1 <= y0 or x1 <= x0: continue
            m = mask[y0 - jy:y1 - jy, x0 - jx:x1 - jx]
            acc[y0:y1, x0:x1] += s[sy + y0 - jy:sy + y1 - jy, sx + x0 - jx:sx + x1 - jx] * m
            ws[y0:y1, x0:x1] += m
    d = acc / np.maximum(ws, 1e-3)
    return (d - d.mean()) / (d.std() + 1e-6)


def maps(key, wcm, hcm, ppc):
    hp, cm = detail_src(key)
    W, H = int(wcm * ppc), int(hcm * ppc)
    det = splat(hp, cm, W, H, ppc)
    target = np.array(SRC[key][3], np.float32); amp = SRC[key][4]
    # distance from edge (cm)
    ey = np.minimum(np.arange(H), np.arange(H)[::-1])[:, None]
    ex = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :]
    e = np.minimum(ey, ex) / ppc
    band = e < 2.5
    # large-scale gentle mottling (pile direction / wear) so the field isn't flat
    lo = splat(hp, cm * 6, W, H, ppc)
    lo = gaussian_filter(lo, ppc * 6)
    lo = lo / (np.abs(lo).max() + 1e-6)
    lum = 1 + amp * np.clip(det, -2.5, 2.5) / 2.5 + 0.02 * lo
    detb = gaussian_filter(det, ppc * 0.3)
    lum = np.where(band, 0.965 + amp * 0.4 * np.clip(detb, -2.5, 2.5) / 2.5, lum)
    seam = np.exp(-((e - 2.5) / 0.12) ** 2); lum *= 1 - 0.10 * seam
    alb = np.clip(target[None, None, :] * lum[..., None], 0, 255).astype(np.uint8)
    # height field -> normal map (pile + binding roll + seam groove)
    h = np.where(band, 0.25 * detb, det) * 0.6
    h += -1.2 * np.clip(1 - e / 0.9, 0, 1) ** 2 - 0.8 * seam
    gy, gx = np.gradient(h)
    k = 1.4
    n = np.dstack([-gx * k, -gy * k, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    nm = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(alb), Image.fromarray(nm)


def glb(wcm, hcm, alb_jpg, nrm_png, side):
    w, d = wcm / 100, hcm / 100; x, z = w / 2, d / 2
    tp = [(-x, TH, -z), (x, TH, -z), (x, TH, z), (-x, TH, z)]
    tn = [(0, 1, 0)] * 4; tuv = [(0, 0), (1, 0), (1, 1), (0, 1)]; tt = [(1, 0, 0, 1)] * 4; ti = [0, 2, 1, 0, 3, 2]
    P, N, I = [], [], []
    def quad(a, b, c, dd, n):
        k0 = len(P); P.extend([a, b, c, dd]); N.extend([n] * 4); I.extend([k0, k0 + 1, k0 + 2, k0, k0 + 2, k0 + 3])
    quad((-x, 0, z), (x, 0, z), (x, TH, z), (-x, TH, z), (0, 0, 1))
    quad((x, 0, -z), (-x, 0, -z), (-x, TH, -z), (x, TH, -z), (0, 0, -1))
    quad((x, 0, z), (x, 0, -z), (x, TH, -z), (x, TH, z), (1, 0, 0))
    quad((-x, 0, -z), (-x, 0, z), (-x, TH, z), (-x, TH, -z), (-1, 0, 0))
    quad((-x, 0, -z), (x, 0, -z), (x, 0, z), (-x, 0, z), (0, -1, 0))
    bufs, views, acc = [], [], []
    def add(data, target=None):
        off = sum(len(b) for b in bufs); bufs.append(data + b'\0' * ((-len(data)) % 4))
        v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
        if target: v['target'] = target
        views.append(v); return len(views) - 1
    def accf(arr, typ, mm=False):
        a = np.array(arr, np.float32); o = {'bufferView': add(a.tobytes(), 34962), 'componentType': 5126, 'count': len(arr), 'type': typ}
        if mm: o['min'] = a.min(0).tolist(); o['max'] = a.max(0).tolist()
        acc.append(o); return len(acc) - 1
    def acci(arr):
        acc.append({'bufferView': add(np.array(arr, np.uint16).tobytes(), 34963), 'componentType': 5123, 'count': len(arr), 'type': 'SCALAR'}); return len(acc) - 1
    a0, a1, a2, a4, a3 = accf(tp, 'VEC3', True), accf(tn, 'VEC3'), accf(tuv, 'VEC2'), accf(tt, 'VEC4'), acci(ti)
    b0, b1, b3 = accf(P, 'VEC3', True), accf(N, 'VEC3'), acci(I)
    iv0 = add(alb_jpg); iv1 = add(nrm_png)
    g = {'asset': {'version': '2.0', 'generator': 'reliv-ar v2'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': 'rug'}],
         'meshes': [{'primitives': [
             {'attributes': {'POSITION': a0, 'NORMAL': a1, 'TEXCOORD_0': a2, 'TANGENT': a4}, 'indices': a3, 'material': 0},
             {'attributes': {'POSITION': b0, 'NORMAL': b1}, 'indices': b3, 'material': 1}]}],
         'materials': [
             {'name': 'pile', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': 0.95},
              'normalTexture': {'index': 1, 'scale': 1.0}},
             {'name': 'side', 'pbrMetallicRoughness': {'baseColorFactor': side, 'metallicFactor': 0, 'roughnessFactor': 1}}],
         'textures': [{'source': 0, 'sampler': 0}, {'source': 1, 'sampler': 0}],
         'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
         'images': [{'bufferView': iv0, 'mimeType': 'image/jpeg'}, {'bufferView': iv1, 'mimeType': 'image/jpeg'}],
         'accessors': acc, 'bufferViews': views}
    binb = b''.join(bufs); g['buffers'] = [{'byteLength': len(binb)}]
    js = json.dumps(g, separators=(',', ':')).encode(); js += b' ' * ((-len(js)) % 4)
    return (struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(binb)) + struct.pack('<II', len(js), 0x4E4F534A) + js
            + struct.pack('<II', len(binb), 0x004E4942) + binb)


def srgb2lin(c): c = np.array(c) / 255; return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

if __name__ == '__main__':
    import os, sys
    OUT = os.path.join(HERE, '..'); os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    cols = sys.argv[1:] or list(SRC)
    for col in cols:
        side = list(srgb2lin(np.array(SRC[col][3]) * 0.8)) + [1]
        for key, w, h in SIZES:
            ppc = min(5.0, 1536 / max(w, h))
            alb, nrm = maps(col, w, h, ppc)
            b1 = io.BytesIO(); alb.save(b1, 'JPEG', quality=88)
            nrm_s = nrm.resize((max(4, nrm.width // 2 * 2), max(4, nrm.height // 2 * 2)))
            b2 = io.BytesIO(); nrm_s.save(b2, 'JPEG', quality=90)
            open(os.path.join(OUT, 'models', f'{col}_{key}.glb'), 'wb').write(glb(w, h, b1.getvalue(), b2.getvalue(), side))
            if key == 'XL':
                alb.resize((alb.width // 2, alb.height // 2)).save(os.path.join(OUT, f'thumb_{col}.jpg'), quality=85)
            print(col, key, alb.size, os.path.getsize(os.path.join(OUT, 'models', f'{col}_{key}.glb')) // 1024, 'KB', flush=True)
