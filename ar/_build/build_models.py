"""v3 rug models — 릴리브 펫슬리브 러그 AR용 GLB 생성.

v2 대비 달라진 점
- 원단 결: 사진에서 결 방향을 자동으로 찾아 러그 긴 변에 맞춰 돌린 뒤, 결 방향으로 길쭉한 조각을 겹쳐 붙여
  헤더(두 가닥 혼방) 줄무늬가 끊기지 않게 함. 결의 진하기는 사진에서 실측한 대비를 그대로 씀(고정값 X).
- 가장자리: 실제 제품처럼 둥근 모서리(r=3cm) + 둥글게 말린 파이핑 테두리(약 1.2cm) + 박음질 홈.
  테두리는 지오메트리(옆면을 1/4 원으로 둥글림)와 노멀맵 둘 다로 표현.
- 노멀맵: 미세 보풀 노이즈를 빼고 파이핑·홈·완만한 굴곡만 넣어 압축 잡티(자글자글)가 안 생기게 함.
- 아이보리: 2025-06 고객 촬영 탑뷰(ivory_top_250625.jpg)의 결을 씀.

사용법: python3 build_models.py [green gray beige ivory]   (색을 안 적으면 전부)
"""
import io, json, os, struct, sys
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, rotate as ndrotate

rng = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, 'src')

# color: (structure photo, crop box, real cm across the crop width, target base sRGB)
SRC = {
    # 2025-10-22 실사(러그색감_AR용) 기준. 색은 흰 가구·종이를 기준으로 화이트밸런스 보정한 중앙값에 AR 조명 감쇠분을 올린 값
    'green': ('green_tex_251022.jpg', (0, 0, 910, 1365), 60.0, (128, 154, 128)),
    'gray':  ('gray_top_251022.jpg', (0, 0, 520, 2048), 20.0, (190, 190, 187)),
    'beige': ('beige_tex_251022.jpg', (515, 0, 966, 483), 25.0, (192, 185, 176)),
    'ivory': ('ivory_tex_251022.jpg', (0, 0, 1365, 1138), 45.0, (224, 219, 208)),
}
SIZES = [('S', 100, 150), ('M', 150, 200), ('L', 200, 250), ('XL', 200, 300),
         ('hall', 100, 250), ('widehall', 120, 200), ('module', 100, 200), ('kitchen', 45, 115)]
TH = 0.009        # 두께 9 mm
CORNER_CM = 1.5   # 모서리 반지름
BAND_CM = 2.5     # 바인딩 테이프 폭(납작한 직조 테두리)
MAX_PX = 1536     # 긴 변 최대 픽셀
PPC_MAX = 5.0     # px / cm 상한


# ---------- 1. 원단 결 추출 ----------
def load_detail(key):
    """사진 -> 조명 제거한 상대 고역(결만), 결 방향을 세로로 정렬. 반환: (detail[H,W] 상대값, cm/px)"""
    f, box, cm, _ = SRC[key]
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC_DIR, f))).convert('L').crop(box)
    a = np.asarray(im).astype(np.float32)
    cm_per_px = cm / a.shape[1]
    low = gaussian_filter(a, 2.0 / cm_per_px)            # 2 cm 보다 큰 명암(조명·주름) 제거
    hp = (a - low) / (low + 1)
    hp = gaussian_filter(hp, 0.12 / cm_per_px)            # 0.1 mm 급 센서 노이즈 억제
    # 결 방향: 구조 텐서로 지배적 방향 찾기, 결이 세로가 되도록 회전(부호는 결과를 보고 고름)
    def streak_angle(a):
        gy, gx = np.gradient(a)
        jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
        return np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90, (jxx - jyy) / (jxx + jyy + 1e-9)
    streak_deg, _ = streak_angle(hp)
    best = None
    for sgn in (1, -1):
        r = ndrotate(hp, sgn * (streak_deg - 90), reshape=False, order=1, mode='reflect')
        H, W = r.shape; m = int(min(H, W) * 0.18)
        r = r[m:H - m, m:W - m]
        _, aniso = streak_angle(r)                         # 세로 결이면 x 방향 변화가 커서 +
        if best is None or aniso > best[0]: best = (aniso, r)
    rot = best[1]
    rms = float(rot.std())
    if rms < 0.02: rot *= 0.02 / rms; rms = 0.02           # 흐린 사진은 최소 가시 대비로 올림
    if rms > 0.09: rot *= 0.09 / rms; rms = 0.09           # 그늘 사진은 과한 대비를 눌러줌
    print(f'  {key}: streak {streak_deg:.1f}deg, aniso {best[0]:.2f}, contrast rms {rms:.3f}', flush=True)
    return rot, cm_per_px


def splat(det, cm_per_px, W, H, ppc, along_v=True):
    """결 방향으로 길쭉한 깃털 조각을 랜덤 위치·뒤집기로 겹쳐 붙여 이음새·반복 없는 결 필드 생성"""
    s = det if along_v else det.T
    scale = cm_per_px * ppc                                # 원본 px -> 출력 px
    sh, sw = max(8, int(s.shape[0] * scale)), max(8, int(s.shape[1] * scale))
    rms0 = s.std()
    s = np.asarray(Image.fromarray(s.astype(np.float32), 'F').resize((sw, sh), Image.LANCZOS))
    s = s * min(2.5, 0.8 * rms0 / (s.std() + 1e-6))         # 축소로 깎인 대비를 80%까지 되살림
    PH, PW = int(sh * 0.8), int(sw * 0.45)                 # 조각: 결 방향으로 길게
    yy, xx = np.mgrid[0:PH, 0:PW]
    mask = np.clip(1 - np.hypot((yy - PH / 2) / (PH / 2), (xx - PW / 2) / (PW / 2)), 0, 1) ** 1.3
    acc = np.zeros((H, W), np.float32); ws = np.zeros((H, W), np.float32)
    sy_, sx_ = max(4, int(PH * 0.42)), max(4, int(PW * 0.42))
    for y in range(-PH // 2, H, sy_):
        for x in range(-PW // 2, W, sx_):
            jy = y + rng.integers(-sy_ // 3, sy_ // 3 + 1); jx = x + rng.integers(-sx_ // 3, sx_ // 3 + 1)
            oy = rng.integers(0, sh - PH + 1); ox = rng.integers(0, sw - PW + 1)
            p = s[oy:oy + PH, ox:ox + PW]
            if rng.random() < .5: p = p[::-1]
            if rng.random() < .5: p = p[:, ::-1]
            y0, x0, y1, x1 = max(jy, 0), max(jx, 0), min(jy + PH, H), min(jx + PW, W)
            if y1 <= y0 or x1 <= x0: continue
            m = mask[y0 - jy:y1 - jy, x0 - jx:x1 - jx]
            acc[y0:y1, x0:x1] += p[y0 - jy:y1 - jy, x0 - jx:x1 - jx] * m
            ws[y0:y1, x0:x1] += m
    return acc / np.maximum(ws, 1e-3)


# ---------- 2. 가장자리 SDF ----------
def edge_distance(W, H, ppc):
    """둥근 사각형 안쪽에서 테두리까지의 거리(cm). 모서리는 원호를 따라감."""
    r = CORNER_CM * ppc
    y = np.arange(H)[:, None] + 0.5; x = np.arange(W)[None, :] + 0.5
    cx, cy = W / 2, H / 2
    qx = np.abs(x - cx) - (W / 2 - r); qy = np.abs(y - cy) - (H / 2 - r)
    sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return -sdf / ppc


# ---------- 3. 텍스처 ----------
def textures(key, wcm, hcm, ppc, det, cm_per_px):
    W, H = int(round(wcm * ppc)), int(round(hcm * ppc))
    target = np.array(SRC[key][3], np.float32)
    along_v = hcm < wcm                                    # 결은 짧은 변 방향(실사 기준: 소파 앞에 놓으면 가로 결)
    d = splat(det, cm_per_px, W, H, ppc, along_v)          # 상대 명암 (실측 대비 유지)
    d = np.clip(d, -0.35, 0.35)
    # 아주 완만한 색 얼룩(보풀 눕는 방향·착용감) — 결 필드를 크게 흐려 재사용
    lo = gaussian_filter(splat(det, cm_per_px * 5, W, H, ppc, along_v), ppc * 8)
    lo = lo / (np.abs(lo).max() + 1e-6)
    e = edge_distance(W, H, ppc)
    band = e < BAND_CM
    lum = 1 + 1.0 * d + 0.03 * lo
    # 바인딩 테이프: 결이 촘촘한 평직. 본체보다 살짝 어둡고(0.955) 가는 가로결
    tape = 0.45 * gaussian_filter(d, 0.35 * ppc)
    weave = 0.025 * np.sin(2 * np.pi * (np.arange(H)[:, None] * (along_v == False) + np.arange(W)[None, :] * (along_v)) / (0.12 * ppc))
    seam_in = np.exp(-((e - BAND_CM) / 0.10) ** 2)         # 안쪽 박음질 홈
    seam_out = np.exp(-((e - 0.35) / 0.08) ** 2)           # 바깥쪽 박음질
    lum = np.where(band, 0.955 + tape + weave, lum) * (1 - 0.12 * seam_in - 0.08 * seam_out)
    stitch = (np.sin(2 * np.pi * (np.arange(W)[None, :] + np.arange(H)[:, None]) / (0.35 * ppc)) > 0.6) * (seam_in + seam_out)
    lum *= 1 + 0.05 * stitch
    alb = np.clip(target[None, None, :] * lum[..., None], 0, 255).astype(np.uint8)
    # 높이: 테이프가 본체보다 아주 살짝 올라오고(두 겹), 박음질 홈 두 줄, 완만한 굴곡. 노이즈 없음
    h = 0.12 * np.clip((BAND_CM - e) / 0.4, 0, 1) * band
    h -= 0.22 * seam_in + 0.12 * seam_out
    h += 0.10 * lo
    h += 0.06 * gaussian_filter(d, 0.5 * ppc) * (~band)
    gy, gx = np.gradient(h)
    k = 5.0 * ppc / 5.0
    n = np.dstack([-gx * k, -gy * k, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    nm = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    return Image.fromarray(alb), Image.fromarray(nm), W, H


# ---------- 4. 지오메트리 ----------
def outline(wcm, hcm, seg=8):
    """둥근 사각형 외곽선 (x,z) 반시계, 단위 m"""
    w, d, r = wcm / 100, hcm / 100, CORNER_CM / 100
    pts = []
    for (cx, cz, a0) in [(w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90), (-w / 2 + r, -d / 2 + r, 180), (w / 2 - r, -d / 2 + r, 270)]:
        for i in range(seg + 1):
            a = np.radians(a0 + 90 * i / seg)
            pts.append((cx + r * np.cos(a), cz + r * np.sin(a)))
    return pts


def mesh(wcm, hcm):
    """상판(둥근 사각형, 선풍기 삼각분할) + 옆면(1/4 원 프로파일) + 바닥. UV는 상판 평면 투영."""
    w, d = wcm / 100, hcm / 100
    ring = outline(wcm, hcm)
    n = len(ring)
    uv = lambda x, z: ((x + w / 2) / w, (z + d / 2) / d)
    P, N, UV, I = [], [], [], []
    # top
    top0 = len(P)
    P.append((0, TH, 0)); N.append((0, 1, 0)); UV.append(uv(0, 0))
    for (x, z) in ring:
        P.append((x, TH, z)); N.append((0, 1, 0)); UV.append(uv(x, z))
    for i in range(n):
        j = (i + 1) % n
        I += [top0, top0 + 1 + j, top0 + 1 + i]           # 위에서 봤을 때 CCW (y-up, 반시계 x->z)
    # side: quarter-round profile from top edge down to the floor edge
    prof = [(np.radians(90 * t / 4)) for t in range(5)]   # 0..90deg
    rows = []
    for a in prof:
        row = []
        inset = TH * (1 - np.sin(a)); y = TH * np.cos(a)  # a=0: top (inset TH), a=90: outer bottom
        for k, (x, z) in enumerate(ring):
            # outward normal in plane from polygon neighbours
            xp, zp = ring[k - 1]; xn, zn = ring[(k + 1) % n]
            tx, tz = xn - xp, zn - zp; L = np.hypot(tx, tz) or 1
            ox, oz = tz / L, -tx / L
            # ring winds CCW viewed from +y (x->z); outward = (tz,-tx) check sign against centre
            if ox * x + oz * z < 0: ox, oz = -ox, -oz
            px, pz = x - ox * inset, z - oz * inset
            nx, ny, nz = ox * np.sin(a), np.cos(a), oz * np.sin(a)
            row.append(len(P)); P.append((px, y, pz)); N.append((nx, ny, nz)); UV.append(uv(px, pz))
        rows.append(row)
    for r in range(len(rows) - 1):
        for k in range(n):
            a, b = rows[r][k], rows[r][(k + 1) % n]
            c, dd = rows[r + 1][k], rows[r + 1][(k + 1) % n]
            I += [a, c, b, b, c, dd]
    # bottom (separate flat-colour material)
    B_P, B_N, B_I = [], [], []
    B_P.append((0, 0, 0)); B_N.append((0, -1, 0))
    for (x, z) in ring: B_P.append((x, 0, z)); B_N.append((0, -1, 0))
    for i in range(n):
        j = (i + 1) % n
        B_I += [0, 1 + i, 1 + j]
    # check winding of top: cross((ring[1]-ring[0]) , ...) ; fix by test
    return P, N, UV, I, B_P, B_N, B_I


def fix_winding(P, I, want_up):
    a, b, c = (np.array(P[I[k]]) for k in range(3))
    nrm = np.cross(b - a, c - a)
    if (nrm[1] > 0) != want_up:
        I = [I[k + off] for k in range(0, len(I), 3) for off in (0, 2, 1)]
    return I


# ---------- 5. GLB ----------
def glb(wcm, hcm, alb_jpg, nrm_jpg, side_rgba, name, rot90=False):
    P, N, UV, I, BP, BN, BI = mesh(wcm, hcm)
    I = fix_winding(P, I, True); BI = fix_winding(BP, BI, False)
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
        a = np.array(arr, np.uint32); acc.append({'bufferView': add(a.tobytes(), 34963), 'componentType': 5125, 'count': len(arr), 'type': 'SCALAR'}); return len(acc) - 1
    tangents = [(1, 0, 0, 1)] * len(P)
    a_p, a_n, a_uv, a_t, a_i = accf(P, 'VEC3', True), accf(N, 'VEC3'), accf(UV, 'VEC2'), accf(tangents, 'VEC4'), acci(I)
    b_p, b_n, b_i = accf(BP, 'VEC3', True), accf(BN, 'VEC3'), acci(BI)
    iv0, iv1 = add(alb_jpg), add(nrm_jpg)
    g = {'asset': {'version': '2.0', 'generator': 'reliv-ar v3'},
         'scene': 0, 'scenes': [{'nodes': [0]}],
         'nodes': [dict({'mesh': 0, 'name': name}, **({'rotation': [0, 0.70710678, 0, 0.70710678]} if rot90 else {}))],
         'meshes': [{'name': name, 'primitives': [
             {'attributes': {'POSITION': a_p, 'NORMAL': a_n, 'TEXCOORD_0': a_uv, 'TANGENT': a_t}, 'indices': a_i, 'material': 0},
             {'attributes': {'POSITION': b_p, 'NORMAL': b_n}, 'indices': b_i, 'material': 1}]}],
         'materials': [
             {'name': 'pile', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': 0.92},
              'normalTexture': {'index': 1, 'scale': 1.0}, 'doubleSided': False},
             {'name': 'backing', 'pbrMetallicRoughness': {'baseColorFactor': side_rgba, 'metallicFactor': 0, 'roughnessFactor': 1}}],
         'textures': [{'source': 0, 'sampler': 0}, {'source': 1, 'sampler': 0}],
         'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
         'images': [{'bufferView': iv0, 'mimeType': 'image/jpeg', 'name': 'albedo'}, {'bufferView': iv1, 'mimeType': 'image/jpeg', 'name': 'normal'}],
         'accessors': acc, 'bufferViews': views}
    binb = b''.join(bufs); g['buffers'] = [{'byteLength': len(binb)}]
    js = json.dumps(g, separators=(',', ':')).encode(); js += b' ' * ((-len(js)) % 4)
    return (struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(binb)) + struct.pack('<II', len(js), 0x4E4F534A) + js
            + struct.pack('<II', len(binb), 0x004E4942) + binb)


def srgb2lin(c):
    c = np.array(c, np.float64) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


if __name__ == '__main__':
    OUT = os.path.join(HERE, '..'); os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    cols = sys.argv[1:] or list(SRC)
    for col in cols:
        det, cmpp = load_detail(col)
        backing = list(srgb2lin((120, 112, 104))) + [1]   # 바닥 논슬립 면: 짙은 회갈색
        for key, w, h in SIZES:
            ppc = min(PPC_MAX, MAX_PX / max(w, h))
            alb, nrm, W, H = textures(col, w, h, ppc, det, cmpp)
            b1 = io.BytesIO(); alb.save(b1, 'JPEG', quality=90, subsampling=0)
            nrm_s = nrm.resize((max(4, W // 2), max(4, H // 2)), Image.LANCZOS)
            b2 = io.BytesIO(); nrm_s.save(b2, 'JPEG', quality=92, subsampling=0)
            path = os.path.join(OUT, 'models', f'{col}_{key}.glb')
            open(path, 'wb').write(glb(w, h, b1.getvalue(), b2.getvalue(), backing, f'reliv_{col}_{key}'))
            # 90도 돌린 판: 안드로이드 Scene Viewer 는 파일 주소만 받으므로 페이지에서 회전을 못 심음
            open(path.replace('.glb', '_r90.glb'), 'wb').write(glb(w, h, b1.getvalue(), b2.getvalue(), backing, f'reliv_{col}_{key}_r90', rot90=True))
            if key == 'XL':
                alb.resize((alb.width // 2, alb.height // 2), Image.LANCZOS).save(os.path.join(OUT, f'thumb_{col}.jpg'), quality=85)
            print(f'{col} {key} {W}x{H} {os.path.getsize(path) // 1024} KB', flush=True)
